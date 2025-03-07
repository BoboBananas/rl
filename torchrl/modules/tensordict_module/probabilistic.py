# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re
from copy import deepcopy
from textwrap import indent
from typing import Sequence, Union, Type, Optional, Tuple

from torch import Tensor
from torch import distributions as d

from torchrl.data import TensorSpec
from torchrl.data.tensordict.tensordict import _TensorDict
from torchrl.envs.utils import exploration_mode, set_exploration_mode
from torchrl.modules.distributions import distributions_maps, Delta
from torchrl.modules.tensordict_module.common import TensorDictModule, _check_all_str

__all__ = ["ProbabilisticTensorDictModule"]


class ProbabilisticTensorDictModule(TensorDictModule):
    """
    A probabilistic TD Module.
    `ProbabilisticTDModule` is a special case of a TDModule where the output is
    sampled given some rule, specified by the input `default_interaction_mode`
    argument and the `exploration_mode()` global function.

    It consists in a wrapper around another TDModule that returns a tensordict
    updated with the distribution parameters. `ProbabilisticTensorDictModule` is
    responsible for constructing the distribution (through the `get_dist()` method)
    and/or sampling from this distribution (through a regular `__call__()` to the
    module).

    A `ProbabilisticTensorDictModule` instance has two main features:
    - It reads and writes TensorDict objects
    - It uses a real mapping R^n -> R^m to create a distribution in R^d from
    which values can be sampled or computed.
    When the `__call__` / `forward` method is called, a distribution is created,
    and a value computed (using the 'mean', 'mode', 'median' attribute or
    the 'rsample', 'sample' method). The sampling step is skipped if the
    inner TDModule has already created the desired key-value pair.

    By default, ProbabilisticTensorDictModule distribution class is a Delta
    distribution, making ProbabilisticTensorDictModule a simple wrapper around
    a deterministic mapping function.

    Args:
        module (nn.Module): a nn.Module used to map the input to the output parameter space. Can be a functional
            module (FunctionalModule or FunctionalModuleWithBuffers), in which case the `forward` method will expect
            the params (and possibly) buffers keyword arguments.
        dist_param_keys (str or iterable of str): key(s) that will be produced
            by the inner TDModule and that will be used to build the distribution.
            Importantly, those keys must match the keywords used by the distribution
            class of interest, e.g. `"loc"` and `"scale"` for the Normal distribution
            and similar.
        out_key_sample (str or iterable of str): keys where the sampled values will be
            written. Importantly, if this key is part of the `out_keys` of the inner model,
            the sampling step will be skipped.
        spec (TensorSpec): specs of the first output tensor. Used when calling td_module.random() to generate random
            values in the target space.
        safe (bool, optional): if True, the value of the sample is checked against the input spec. Out-of-domain sampling can
            occur because of exploration policies or numerical under/overflow issues. As for the `spec` argument,
            this check will only occur for the distribution sample, but not the other tensors returned by the input
            module. If the sample is out of bounds, it is projected back onto the desired space using the
            `TensorSpec.project`
            method.
            Default is `False`.
        default_interaction_mode (str, optional): default method to be used to retrieve the output value. Should be one of:
            'mode', 'median', 'mean' or 'random' (in which case the value is sampled randomly from the distribution).
            Default is 'mode'.
            Note: When a sample is drawn, the `ProbabilisticTDModule` instance will fist look for the interaction mode
            dictated by the `exploration_mode()` global function. If this returns `None` (its default value),
            then the `default_interaction_mode` of the `ProbabilisticTDModule` instance will be used.
            Note that DataCollector instances will use `set_exploration_mode` to `"random"` by default.
        distribution_class (Type, optional): a torch.distributions.Distribution class to be used for sampling.
            Default is Delta.
        distribution_kwargs (dict, optional): kwargs to be passed to the distribution.
        return_log_prob (bool, optional): if True, the log-probability of the distribution sample will be written in the
            tensordict with the key `f'{in_keys[0]}_log_prob'`. Default is `False`.
        cache_dist (bool, optional): EXPERIMENTAL: if True, the parameters of the distribution (i.e. the output of the module)
            will be written to the tensordict along with the sample. Those parameters can be used to
            re-compute the original distribution later on (e.g. to compute the divergence between the distribution
            used to sample the action and the updated distribution in PPO).
            Default is `False`.
        n_empirical_estimate (int, optional): number of samples to compute the empirical mean when it is not available.
            Default is 1000

    Examples:
        >>> from torchrl.modules.td_module import ProbabilisticTensorDictModule
        >>> from torchrl.data import TensorDict, NdUnboundedContinuousTensorSpec
        >>> from torchrl.modules import  TanhNormal, NormalParamWrapper
        >>> import functorch, torch
        >>> td = TensorDict({"input": torch.randn(3, 4), "hidden": torch.randn(3, 8)}, [3,])
        >>> spec = NdUnboundedContinuousTensorSpec(4)
        >>> net = NormalParamWrapper(torch.nn.GRUCell(4, 8))
        >>> fnet, params, buffers = functorch.make_functional_with_buffers(net)
        >>> module = TensorDictModule(fnet, in_keys=["input", "hidden"], out_keys=["loc", "scale"])
        >>> td_module = ProbabilisticTensorDictModule(
        ...    module=module,
        ...    spec=spec,
        ...    dist_param_keys=["loc", "scale"],
        ...    out_key_sample=["action"],
        ...    distribution_class=TanhNormal,
        ...    return_log_prob=True,
        ...    )
        >>> _ = td_module(td, params=params, buffers=buffers)
        >>> print(td)
        TensorDict(
            fields={
                input: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                hidden: Tensor(torch.Size([3, 8]), dtype=torch.float32),
                loc: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                scale: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                action: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                sample_log_prob: Tensor(torch.Size([3, 1]), dtype=torch.float32)},
            batch_size=torch.Size([3]),
            device=cpu,
            is_shared=False)

        >>> # In the vmap case, the tensordict is again expended to match the batch:
        >>> params = tuple(p.expand(4, *p.shape).contiguous().normal_() for p in params)
        >>> buffers = tuple(b.expand(4, *b.shape).contiguous().normal_() for p in buffers)
        >>> td_vmap = td_module(td, params=params, buffers=buffers, vmap=True)
        >>> print(td_vmap)
        TensorDict(
            fields={
                input: Tensor(torch.Size([4, 3, 4]), dtype=torch.float32),
                hidden: Tensor(torch.Size([4, 3, 8]), dtype=torch.float32),
                loc: Tensor(torch.Size([4, 3, 4]), dtype=torch.float32),
                scale: Tensor(torch.Size([4, 3, 4]), dtype=torch.float32),
                action: Tensor(torch.Size([4, 3, 4]), dtype=torch.float32),
                sample_log_prob: Tensor(torch.Size([4, 3, 1]), dtype=torch.float32)},
            batch_size=torch.Size([4, 3]),
            device=cpu,
            is_shared=False)

    """

    def __init__(
        self,
        module: TensorDictModule,
        dist_param_keys: Union[str, Sequence[str]],
        out_key_sample: Union[str, Sequence[str]],
        spec: Optional[TensorSpec] = None,
        safe: bool = False,
        default_interaction_mode: str = "mode",
        distribution_class: Type = Delta,
        distribution_kwargs: Optional[dict] = None,
        return_log_prob: bool = False,
        cache_dist: bool = False,
        n_empirical_estimate: int = 1000,
    ):
        in_keys = module.in_keys

        # if the module returns the sampled key we wont be sampling it again
        # then ProbabilisticTensorDictModule is presumably used to return the distribution using `get_dist`
        if isinstance(dist_param_keys, str):
            dist_param_keys = [dist_param_keys]
        if isinstance(out_key_sample, str):
            out_key_sample = [out_key_sample]
        for key in dist_param_keys:
            if key not in module.out_keys:
                raise RuntimeError(
                    f"The key {key} could not be found in the wrapped module `{type(module)}.out_keys`."
                )
        module_out_keys = module.out_keys
        self.out_key_sample = out_key_sample
        _check_all_str(self.out_key_sample)
        out_key_sample = [key for key in out_key_sample if key not in module_out_keys]
        self._requires_sample = bool(len(out_key_sample))
        out_keys = out_key_sample + module_out_keys
        super().__init__(
            module=module, spec=spec, in_keys=in_keys, out_keys=out_keys, safe=safe
        )
        self.dist_param_keys = dist_param_keys
        _check_all_str(self.dist_param_keys)

        self.default_interaction_mode = default_interaction_mode
        if isinstance(distribution_class, str):
            distribution_class = distributions_maps.get(distribution_class.lower())
        self.distribution_class = distribution_class
        self.distribution_kwargs = (
            distribution_kwargs if distribution_kwargs is not None else dict()
        )
        self.n_empirical_estimate = n_empirical_estimate
        self._dist = None
        self.cache_dist = cache_dist if hasattr(distribution_class, "update") else False
        self.return_log_prob = return_log_prob

    def _call_module(self, tensordict: _TensorDict, **kwargs) -> _TensorDict:
        return self.module(tensordict, **kwargs)

    def make_functional_with_buffers(self, clone: bool = True):
        module_params = self.parameters(recurse=False)
        if len(list(module_params)):
            raise RuntimeError(
                "make_functional_with_buffers cannot be called on ProbabilisticTensorDictModule"
                "that contain parameters on the outer level."
            )
        if clone:
            self_copy = deepcopy(self)
        else:
            self_copy = self

        self_copy.module, other = self_copy.module.make_functional_with_buffers(
            clone=True
        )
        return self_copy, other

    def get_dist(
        self,
        tensordict: _TensorDict,
        tensordict_out: Optional[_TensorDict] = None,
        **kwargs,
    ) -> Tuple[d.Distribution, _TensorDict]:
        interaction_mode = exploration_mode()
        if interaction_mode is None:
            interaction_mode = self.default_interaction_mode
        with set_exploration_mode(interaction_mode):
            tensordict_out = self._call_module(
                tensordict, tensordict_out=tensordict_out, **kwargs
            )
        dist = self.build_dist_from_params(tensordict_out)
        return dist, tensordict_out

    def build_dist_from_params(self, tensordict_out: _TensorDict) -> d.Distribution:
        try:
            dist = self.distribution_class(
                **tensordict_out.select(*self.dist_param_keys)
            )
        except TypeError as err:
            if "an unexpected keyword argument" in str(err):
                raise TypeError(
                    "distribution keywords and tensordict keys indicated by ProbabilisticTensorDictModule.dist_param_keys must match."
                    f"Got this error message: \n{indent(str(err), 4 * ' ')}\nwith dist_param_keys={self.dist_param_keys}"
                )
            elif re.search(r"missing.*required positional arguments", str(err)):
                raise TypeError(
                    f"TensorDict with keys {tensordict_out.keys()} does not match the distribution {self.distribution_class} keywords."
                )
            else:
                raise err
        return dist

    def forward(
        self,
        tensordict: _TensorDict,
        tensordict_out: Optional[_TensorDict] = None,
        **kwargs,
    ) -> _TensorDict:

        dist, tensordict_out = self.get_dist(
            tensordict, tensordict_out=tensordict_out, **kwargs
        )
        if self._requires_sample:
            out_tensors = self._dist_sample(dist, interaction_mode=exploration_mode())
            if isinstance(out_tensors, Tensor):
                out_tensors = (out_tensors,)
            tensordict_out.update(
                {key: value for key, value in zip(self.out_key_sample, out_tensors)}
            )
            if self.return_log_prob:
                log_prob = dist.log_prob(*out_tensors)
                tensordict_out.set("sample_log_prob", log_prob)
        elif self.return_log_prob:
            out_tensors = [tensordict_out.get(key) for key in self.out_key_sample]
            log_prob = dist.log_prob(*out_tensors)
            tensordict_out.set("sample_log_prob", log_prob)
            # raise RuntimeError(
            #     "ProbabilisticTensorDictModule.return_log_prob = True is incompatible with settings in which "
            #     "the submodule is responsible for sampling. To manually gather the log-probability, call first "
            #     "\n>>> dist, tensordict = tensordict_module.get_dist(tensordict)"
            #     "\n>>> tensordict.set('sample_log_prob', dist.log_prob(tensordict.get(sample_key))"
            # )
        return tensordict_out

    def _dist_sample(
        self,
        dist: d.Distribution,
        *tensors: Tensor,
        interaction_mode: bool = None,
    ) -> Union[Tuple[Tensor], Tensor]:
        if interaction_mode is None:
            interaction_mode = self.default_interaction_mode
        if not isinstance(dist, d.Distribution):
            raise TypeError(f"type {type(dist)} not recognised by _dist_sample")

        if interaction_mode == "mode":
            if hasattr(dist, "mode"):
                return dist.mode
            else:
                raise NotImplementedError(
                    f"method {type(dist)}.mode is not implemented"
                )

        elif interaction_mode == "median":
            if hasattr(dist, "median"):
                return dist.median
            else:
                raise NotImplementedError(
                    f"method {type(dist)}.median is not implemented"
                )

        elif interaction_mode == "mean":
            try:
                return dist.mean
            except AttributeError:
                if dist.has_rsample:
                    return dist.rsample((self.n_empirical_estimate,)).mean(0)
                else:
                    return dist.sample((self.n_empirical_estimate,)).mean(0)

        elif interaction_mode == "random":
            if dist.has_rsample:
                return dist.rsample()
            else:
                return dist.sample()
        else:
            raise NotImplementedError(f"unknown interaction_mode {interaction_mode}")

    @property
    def num_params(self):
        return self.module.num_params

    @property
    def num_buffers(self):
        return self.module.num_buffers
