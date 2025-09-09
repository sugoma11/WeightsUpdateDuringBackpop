import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class MyLinearFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, lr=0, prob_of_using_new=0):
        ctx.save_for_backward(input, weight, bias)

        ctx.lr = lr
        ctx.prob_of_using_new = prob_of_using_new

        output = F.linear(input, weight, bias)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors

        lr = ctx.lr
        prob_of_using_new = ctx.prob_of_using_new

        grad_input = grad_weight = grad_bias = None


        if ctx.needs_input_grad[1]:
            # For batched inputs
            if input.dim() > 2:
                # Reshape for matrix multiplication
                input_reshaped = input.reshape(-1, input.size(-1))
                grad_output_reshaped = grad_output.reshape(-1, grad_output.size(-1))

                # Properly transpose dimensions for weight gradient
                grad_weight = grad_output_reshaped.t().matmul(input_reshaped)
            else:
                grad_weight = grad_output.t().matmul(input)

        if ctx.needs_input_grad[0]:

            use_new = np.random.binomial(1, float(prob_of_using_new)) == 1

            if use_new:
                weight = weight - lr * grad_weight

            grad_input = F.linear(grad_output, weight.t())

        if bias is not None and ctx.needs_input_grad[2]:
            # Sum across all dimensions except the last
            dims = tuple(range(grad_output.dim() - 1))
            grad_bias = grad_output.sum(dims)

        return grad_input, grad_weight, grad_bias, None, None


class MyLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super(MyLinear, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.in_features = in_features
        self.out_features = out_features

        # Create the same parameters as nn.Linear
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        # Use the same initialization as nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input, lr, prob_of_using_new):
        return MyLinearFunction.apply(input, self.weight, self.bias, lr, prob_of_using_new)

    def extra_repr(self):
        # Match the representation string of nn.Linear
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

    @classmethod
    def from_linear(cls, linear):
        """Create a MyLinear layer from an existing nn.Linear layer, copying its weights and bias."""
        if not isinstance(linear, nn.Linear):
            raise TypeError("Expected nn.Linear instance, got {}".format(type(linear)))

        my_linear = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype
        )

        # Copy weights and bias
        with torch.no_grad():
            my_linear.weight.copy_(linear.weight)
            if linear.bias is not None and my_linear.bias is not None:
                my_linear.bias.copy_(linear.bias)

        return my_linear

    def load_weights(self, weight, bias=None):
        """Load weights and optional bias into this layer."""
        if weight.size() != self.weight.size():
            raise ValueError(f"Weight size mismatch: expected {self.weight.size()}, got {weight.size()}")

        with torch.no_grad():
            self.weight.copy_(weight)

        if bias is not None:
            if self.bias is None:
                raise ValueError("Cannot load bias when layer was initialized with bias=False")
            if bias.size() != self.bias.size():
                raise ValueError(f"Bias size mismatch: expected {self.bias.size()}, got {bias.size()}")
            with torch.no_grad():
                self.bias.copy_(bias)

        return self


class My1DBatchNormFunction(Function):
    @staticmethod
    def forward(ctx, input, running_mean, running_var, weight, bias,
                training=True, momentum=0.1, eps=1e-5, lr=0, prob_of_using_new=0):
        # Save for backward
        ctx.save_for_backward(input, weight, bias, running_mean, running_var)
        ctx.eps = eps
        ctx.momentum = momentum
        ctx.training = training

        ctx.lr = lr
        ctx.prob_of_using_new = prob_of_using_new

        if training:
            # Calculate batch statistics
            batch_size = input.size(0)

            # Dimensions to reduce over: (0, 2, 3) for 4D input (batch, channels, height, width)
            # or (0) for 2D input (batch, features)
            reduce_dims = [0]
            if input.dim() > 2:
                reduce_dims.extend(list(range(2, input.dim())))

            batch_mean = input.mean(dim=reduce_dims)
            batch_var = input.var(dim=reduce_dims, unbiased=False)

            # Update running statistics
            if running_mean is not None and running_var is not None:
                with torch.no_grad():
                    running_mean.copy_((1 - momentum) * running_mean + momentum * batch_mean)
                    running_var.copy_((1 - momentum) * running_var + momentum * batch_var)

            n = input.numel() / input.size(1)
            input_normalized = (input - batch_mean.view(-1, *([1] * (input.dim() - 2)))) / torch.sqrt(batch_var.view(-1, *([1] * (input.dim() - 2))) + eps)
        else:
            # Use running statistics
            input_normalized = (input - running_mean.view(-1, *([1] * (input.dim() - 2)))) / torch.sqrt(running_var.view(-1, *([1] * (input.dim() - 2))) + eps)

        # Apply affine transform
        if weight is not None and bias is not None:
            output = weight.view(-1, *([1] * (input.dim() - 2))) * input_normalized + bias.view(-1, *([1] * (input.dim() - 2)))
        else:
            output = input_normalized

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, running_mean, running_var = ctx.saved_tensors
        eps = ctx.eps
        training = ctx.training

        lr = ctx.lr
        prob_of_using_new = ctx.prob_of_using_new

        grad_input = grad_weight = grad_bias = grad_running_mean = grad_running_var = None
        grad_momentum = grad_eps = None

        # Dimensions to reduce over
        reduce_dims = [0]
        if input.dim() > 2:
            reduce_dims.extend(list(range(2, input.dim())))

        if training:
            batch_mean = input.mean(dim=reduce_dims)
            batch_var = input.var(dim=reduce_dims, unbiased=False)
        else:
            batch_mean = running_mean
            batch_var = running_var

        if ctx.needs_input_grad[3] and weight is not None:
            x_norm = (input - batch_mean.view(-1, *([1] * (input.dim() - 2)))) / torch.sqrt(batch_var.view(-1, *([1] * (input.dim() - 2))) + eps)
            grad_weight = (grad_output * x_norm).sum(dim=reduce_dims)

            use_new = np.random.binomial(1, float(prob_of_using_new)) == 1
            if use_new:
                weight = weight - lr * grad_weight


        if ctx.needs_input_grad[0]:
            # Calculate gradients for input
            N = input.numel() / input.size(1)
            std = torch.sqrt(batch_var.view(-1, *([1] * (input.dim() - 2))) + eps)

            # Compute input gradient
            weight_expanded = weight.view(-1, *([1] * (input.dim() - 2))) if weight is not None else 1

            # Gradients from standard BatchNorm formulation
            x_norm = (input - batch_mean.view(-1, *([1] * (input.dim() - 2)))) / std

            # Compute gradient with respect to input
            dx_norm = grad_output * weight_expanded

            # Formula for input gradient
            grad_input = (N * dx_norm - dx_norm.sum(dim=reduce_dims, keepdim=True) -
                          x_norm * (dx_norm * x_norm).sum(dim=reduce_dims, keepdim=True)) / (N * std)

        if ctx.needs_input_grad[4] and bias is not None:
            # Gradient for bias
            grad_bias = grad_output.sum(dim=reduce_dims)

        # Running stats don't need gradients
        return grad_input, None, None, grad_weight, grad_bias, None, None, None, None, None


class My1DBatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, device=None, dtype=None):
        super(My1DBatchNorm, self).__init__()
        factory_kwargs = {}
        if device is not None:
            factory_kwargs['device'] = device
        if dtype is not None:
            factory_kwargs['dtype'] = dtype

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features, **factory_kwargs))
            self.bias = nn.Parameter(torch.zeros(num_features, **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features, **factory_kwargs))
            self.register_buffer('running_var', torch.ones(num_features, **factory_kwargs))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long, **factory_kwargs))
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)
            self.register_buffer('num_batches_tracked', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input, lr, prob_of_using_new):
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # Use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)

        return My1DBatchNormFunction.apply(
            input,
            self.running_mean if self.track_running_stats else None,
            self.running_var if self.track_running_stats else None,
            self.weight if self.affine else None,
            self.bias if self.affine else None,
            self.training,
            exponential_average_factor,
            self.eps,
            lr, prob_of_using_new
        )

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

    @classmethod
    def from_bn(cls, bn):
        """Create a My1DBatchNorm layer from an existing nn.BatchNorm1d layer, copying its parameters."""
        if not isinstance(bn, nn.BatchNorm1d):
            raise TypeError("Expected nn.BatchNorm1d instance, got {}".format(type(bn)))

        my_bn = cls(
            bn.num_features,
            bn.eps,
            bn.momentum,
            bn.affine,
            bn.track_running_stats,
            device=bn.weight.device if bn.affine else None,
            # dtype=bn.weight.dtype if bn.affine else None
        )

        # Copy parameters and buffers
        with torch.no_grad():
            if bn.affine:
                my_bn.weight.copy_(bn.weight)
                my_bn.bias.copy_(bn.bias)

            if bn.track_running_stats:
                my_bn.running_mean.copy_(bn.running_mean)
                my_bn.running_var.copy_(bn.running_var)
                my_bn.num_batches_tracked.copy_(bn.num_batches_tracked)

        return my_bn

    def load_parameters(self, weight=None, bias=None, running_mean=None, running_var=None):
        """Load parameters into this batch norm layer."""
        if weight is not None:
            if not self.affine:
                raise ValueError("Cannot load weight when affine=False")
            if weight.size() != self.weight.size():
                raise ValueError(f"Weight size mismatch: expected {self.weight.size()}, got {weight.size()}")
            with torch.no_grad():
                self.weight.copy_(weight)

        if bias is not None:
            if not self.affine:
                raise ValueError("Cannot load bias when affine=False")
            if bias.size() != self.bias.size():
                raise ValueError(f"Bias size mismatch: expected {self.bias.size()}, got {bias.size()}")
            with torch.no_grad():
                self.bias.copy_(bias)

        if running_mean is not None:
            if not self.track_running_stats:
                raise ValueError("Cannot load running_mean when track_running_stats=False")
            if running_mean.size() != self.running_mean.size():
                raise ValueError(f"Running mean size mismatch: expected {self.running_mean.size()}, got {running_mean.size()}")
            with torch.no_grad():
                self.running_mean.copy_(running_mean)

        if running_var is not None:
            if not self.track_running_stats:
                raise ValueError("Cannot load running_var when track_running_stats=False")
            if running_var.size() != self.running_var.size():
                raise ValueError(f"Running var size mismatch: expected {self.running_var.size()}, got {running_var.size()}")
            with torch.no_grad():
                self.running_var.copy_(running_var)

        return self



class CustomSequential(nn.Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x, **kwargs):
        for layer in self.layers:
            if isinstance(layer, (MyLinear, My1DBatchNorm)):
                x = layer(x, **kwargs)
            else:
                x = layer(x)
        return x


def get_fc_network(input_dim, num_layers, use_bn, feature_pass, num_classes, device):

    layers_old_w = []
    layers_new_w = []

    fan_in = input_dim

    for _ in range(num_layers - 1):

        if feature_pass == 'triangle':
            fan_out = int(2 ** np.floor(np.log2(fan_in) - 1))

        elif feature_pass == 'persisting':
            fan_out = fan_in

        old_w_layer = nn.Linear(fan_in, fan_out)
        layers_old_w.append(MyLinear.from_linear(old_w_layer))
        layers_new_w.append(MyLinear.from_linear(old_w_layer))

        if use_bn:
            layers_old_w.append(My1DBatchNorm(fan_out))
            layers_new_w.append(My1DBatchNorm(fan_out))

        layers_old_w.append(nn.ReLU())
        layers_new_w.append(nn.ReLU())

        fan_in = fan_out

    last_layer = nn.Linear(fan_out, num_classes)

    layers_old_w.append(MyLinear.from_linear(last_layer))
    layers_new_w.append(MyLinear.from_linear(last_layer))

    return CustomSequential(*layers_old_w).to(device), CustomSequential(*layers_new_w).to(device)


if __name__ == '__main__':
    b = nn.BatchNorm1d(10)
    t = My1DBatchNorm.from_bn(b)