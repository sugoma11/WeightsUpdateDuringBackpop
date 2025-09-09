import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from core.layers import MyLinear, My1DBatchNorm, CustomSequential


class TestForwardPass:
    """Test forward pass consistency between networks."""

    def test_forward_pass_similarity(self, networks, sample_input):
        """Test that both networks produce approximately the same output during forward pass."""
        old_net, new_net = networks

        # Set both networks to eval mode to avoid BatchNorm training behavior
        old_net.eval()
        new_net.eval()

        # Forward pass with prob_of_using_new=0 (no weight updates during backward)
        with torch.no_grad():
            output_old = old_net(sample_input, lr=0.01, prob_of_using_new=0.0)
            output_new = new_net(sample_input, lr=0.01, prob_of_using_new=0.0)

        # Check that outputs are approximately equal
        max_diff = torch.max(torch.abs(output_old - output_new)).item()
        mean_diff = torch.mean(torch.abs(output_old - output_new)).item()

        print(f"Forward pass - Max difference: {max_diff:.6e}")
        print(f"Forward pass - Mean difference: {mean_diff:.6e}")

        # Allow for small numerical errors (due to floating point precision)
        assert torch.allclose(output_old, output_new, rtol=1e-5, atol=1e-6), \
            f"Outputs differ too much. Max diff: {max_diff}, Mean diff: {mean_diff}"

    def test_forward_pass_training_mode(self, networks, sample_input):
        """Test forward pass in training mode with BatchNorm."""
        old_net, new_net = networks

        # Set to training mode
        old_net.train()
        new_net.train()

        # Forward pass
        output_old = old_net(sample_input, lr=0.01, prob_of_using_new=0.0)
        output_new = new_net(sample_input, lr=0.01, prob_of_using_new=0.0)

        # In training mode with same initialization, outputs should still be very similar
        assert torch.allclose(output_old, output_new, rtol=1e-4, atol=1e-5), \
            "Outputs differ too much in training mode"


class TestGradients:
    """Test gradient behavior with different prob_of_using_new values."""

    def test_gradients_prob_zero(self, networks, sample_input, sample_target):
        """Test that gradients are similar when prob_of_using_new=0."""
        old_net, new_net = networks

        # Set to training mode
        old_net.train()
        new_net.train()

        # Reset gradients
        old_net.zero_grad()
        new_net.zero_grad()

        # Forward pass with prob_of_using_new=0
        output_old = old_net(sample_input, lr=0.01, prob_of_using_new=0.0)
        output_new = new_net(sample_input, lr=0.01, prob_of_using_new=0.0)

        # Compute loss
        loss_old = F.cross_entropy(output_old, sample_target)
        loss_new = F.cross_entropy(output_new, sample_target)

        # Backward pass
        loss_old.backward()
        loss_new.backward()

        # Get first layer (should be MyLinear)
        first_layer_old = None
        first_layer_new = None

        for layer in old_net.layers:
            if isinstance(layer, MyLinear):
                first_layer_old = layer
                break

        for layer in new_net.layers:
            if isinstance(layer, MyLinear):
                first_layer_new = layer
                break

        assert first_layer_old is not None and first_layer_new is not None, \
            "Could not find first MyLinear layer"

        # Check weight gradients
        grad_old = first_layer_old.weight.grad
        grad_new = first_layer_new.weight.grad

        assert grad_old is not None and grad_new is not None, \
            "Gradients were not computed"

        # Calculate differences
        max_diff = torch.max(torch.abs(grad_old - grad_new)).item()
        mean_diff = torch.mean(torch.abs(grad_old - grad_new)).item()
        relative_diff = torch.norm(grad_old - grad_new) / torch.norm(grad_old)

        print(f"\nProb=0.0 - Weight gradient max difference: {max_diff:.6e}")
        print(f"Prob=0.0 - Weight gradient mean difference: {mean_diff:.6e}")
        print(f"Prob=0.0 - Weight gradient relative difference: {relative_diff:.6e}")

        # With prob_of_using_new=0, gradients should be very similar
        assert torch.allclose(grad_old, grad_new, rtol=1e-4, atol=1e-5), \
            f"Gradients differ too much with prob=0. Max diff: {max_diff}, Relative diff: {relative_diff}"

    def test_gradients_prob_one(self, networks, sample_input, sample_target):
        """Test that gradients differ significantly when prob_of_using_new=1."""
        old_net, new_net = networks

        # Set to training mode
        old_net.train()
        new_net.train()

        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)

        # Reset gradients
        old_net.zero_grad()
        new_net.zero_grad()

        # Forward pass with different probabilities
        output_old = old_net(sample_input, lr=0.1, prob_of_using_new=0.0)
        output_new = new_net(sample_input, lr=0.1, prob_of_using_new=1.0)

        # Compute loss
        loss_old = F.cross_entropy(output_old, sample_target)
        loss_new = F.cross_entropy(output_new, sample_target)

        # Backward pass
        loss_old.backward()
        loss_new.backward()

        # Get first layer
        first_layer_old = None
        first_layer_new = None

        for layer in old_net.layers:
            if isinstance(layer, MyLinear):
                first_layer_old = layer
                break

        for layer in new_net.layers:
            if isinstance(layer, MyLinear):
                first_layer_new = layer
                break

        assert first_layer_old is not None and first_layer_new is not None, \
            "Could not find first MyLinear layer"

        # Check weight gradients
        grad_old = first_layer_old.weight.grad
        grad_new = first_layer_new.weight.grad

        assert grad_old is not None and grad_new is not None, \
            "Gradients were not computed"

        # Calculate differences
        max_diff = torch.max(torch.abs(grad_old - grad_new)).item()
        mean_diff = torch.mean(torch.abs(grad_old - grad_new)).item()
        relative_diff = torch.norm(grad_old - grad_new) / torch.norm(grad_old)

        print(f"\nProb=1.0 - Weight gradient max difference: {max_diff:.6e}")
        print(f"Prob=1.0 - Weight gradient mean difference: {mean_diff:.6e}")
        print(f"Prob=1.0 - Weight gradient relative difference: {relative_diff:.6e}")

        # With prob_of_using_new=1, gradients should differ significantly
        # The threshold here depends on the learning rate and network depth
        assert relative_diff > 0.01, \
            f"Gradients should differ significantly with prob=1.0. Relative diff: {relative_diff:.6e}"

        # Also check that they're not completely unrelated (sanity check)
        assert relative_diff < 10.0, \
            f"Gradients differ too much (might indicate a bug). Relative diff: {relative_diff:.6e}"


class TestMyLinear:
    """Test MyLinear module functionality."""

    def test_from_linear_conversion(self, device):
        """Test converting nn.Linear to MyLinear."""
        # Create original linear layer
        original = nn.Linear(64, 32, bias=True).to(device)

        # Convert to MyLinear
        my_linear = MyLinear.from_linear(original)

        # Check dimensions
        assert my_linear.in_features == original.in_features
        assert my_linear.out_features == original.out_features

        # Check weights are copied correctly
        assert torch.allclose(my_linear.weight, original.weight)
        assert torch.allclose(my_linear.bias, original.bias)

        # Test forward pass
        x = torch.randn(16, 64, device=device)
        with torch.no_grad():
            out_original = original(x)
            out_my = my_linear(x, lr=0.01, prob_of_using_new=0.0)

        assert torch.allclose(out_original, out_my, rtol=1e-5, atol=1e-6)


class TestMy1DBatchNorm:
    """Test My1DBatchNorm module functionality."""

    def test_from_bn_conversion(self, device):
        """Test converting nn.BatchNorm1d to My1DBatchNorm."""
        # Create original batch norm
        original = nn.BatchNorm1d(64).to(device)

        # Convert to My1DBatchNorm
        my_bn = My1DBatchNorm.from_bn(original)

        # Check parameters
        assert my_bn.num_features == original.num_features
        assert my_bn.eps == original.eps
        assert my_bn.momentum == original.momentum

        # Check weights are copied correctly
        assert torch.allclose(my_bn.weight, original.weight)
        assert torch.allclose(my_bn.bias, original.bias)
        assert torch.allclose(my_bn.running_mean, original.running_mean)
        assert torch.allclose(my_bn.running_var, original.running_var)

        # Test forward pass in eval mode
        original.eval()
        my_bn.eval()

        x = torch.randn(16, 64, device=device)
        with torch.no_grad():
            out_original = original(x)
            out_my = my_bn(x, lr=0.01, prob_of_using_new=0.0)

        assert torch.allclose(out_original, out_my, rtol=1e-4, atol=1e-5)


class TestCustomSequential:
    """Test CustomSequential module."""

    def test_mixed_layers(self, device):
        """Test CustomSequential with mixed layer types."""
        layers = [
            MyLinear(64, 32),
            My1DBatchNorm(32),
            nn.ReLU(),
            MyLinear(32, 16),
            nn.ReLU()
        ]

        model = CustomSequential(*layers).to(device)

        # Test forward pass
        x = torch.randn(8, 64, device=device)
        output = model(x, lr=0.01, prob_of_using_new=0.5)

        assert output.shape == (8, 16)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()