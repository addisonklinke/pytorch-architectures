from abc import ABC, abstractmethod
import torch


class DimensionSize(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def matches(self, value):
        """To be defined by the derived class"""
        raise NotImplementedError


class AnyDimension(DimensionSize):

    def __init__(self):
        super(AnyDimension, self).__init__()

    def matches(self, value):
        return True


class DimensionRange(DimensionSize):

    def __init__(self, low, high):
        super(DimensionRange, self).__init__()
        self.low = low
        self.high = high

    def matches(self, value):
        return self.low <= value <= self.high


class FixedDimension(DimensionSize):

    def __init__(self, fixed):
        super(FixedDimension, self).__init__()
        self.fixed = fixed

    def matches(self, value):
        return value == self.fixed


class MaxDimension(DimensionSize):

    def __init__(self, maximum):
        super(MaxDimension, self).__init__()
        self.maximum = maximum

    def matches(self, value):
        return value <= self.maximum


class MinDimension(DimensionSize):

    def __init__(self, minimum):
        super(MinDimension, self).__init__()
        self.minimum = minimum

    def matches(self, value):
        return self.minimum <= value


class TensorSpec:

    def __init__(self, schema, dtype=torch.float32, device='cpu'):
        """Initialize a tensor specification

        :param int or list[DimensionSize] schema: Length defines the rank
            (i.e. number of dimensions) of the tensor. Any provided
            ``DimensionSize`` elements define the specifics of each dimension.
            For instance, the spec for an ImageNet 224px square batch of
            images would...

                * Follow ``(N, C, W, H)`` convention
                * Allow any batch size
                * Require 3 channels
                * Require 224 pixel width and height

            Translating this into ``DimensionSize`` elements would look like

                [
                    AnyDimension(),
                    FixedDimension(3),
                    FixedDimension(224),
                    FixedDimension(224),
                ]

        :param torch.dtype dtype: Required data type of the tensor
        :param str or torch.device device: Device tensor should be on
        """

        # Check dtype and device
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f'dtype must be torch.dtype, got {type(dtype)}')
        self.dtype = dtype

        if isinstance(device, torch.device):
            self.device = device
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            raise ValueError(f'device must be torch.device or str, got {type(device)}')

        # Determine schema format
        if isinstance(schema, int):
            self.rank = schema
            self.validators = []
        elif isinstance(schema, list):
            if not all([isinstance(item, DimensionSize) for item in schema]):
                raise ValueError('schema of type list must contain only DimensionSize elements')
            self.rank = len(schema)
            self.validators = schema
        else:
            raise NotImplementedError(f'schema must be int or list[DimensionSize], got {type(schema)}')

    def matches(self, other, require_dtype=True, require_device=True):
        """Determine whether a 3rd party tensor matches this class' spec

        :param torch.tensor other: To validate
        :param bool require_dtype: Whether to require the same dtype
        :param bool require_device: Whether to require the same device
        :return bool: Whether or not all spec items match
        """

        # Start with simple disqualifications based on type, shape, and/or device
        if not isinstance(other, torch.Tensor):
            raise ValueError(f'other must be Tensor, got {type(other)}')
        if require_dtype and other.dtype != self.dtype:
            return False
        if require_device and other.device != self.device:
            return False
        if len(other.shape) != self.rank:
            return False

        # Detailed validation for each dimension
        is_match = True
        for dim, val in zip(other.shape, self.validators):
            if not val.matches(dim):
                is_match = False
                break
        return is_match
