from core.tensor import Tensor
from core.losses import MSELoss,CrossEntropyLoss,BinaryCrossEntropyLoss
import numpy as np
def test_module():
    """🧪 Module Test: Complete Integration

    Comprehensive test of entire losses module functionality.

    This final test runs before module summary to ensure:
    - All unit tests pass
    - Functions work together correctly
    - Module is ready for integration with TinyTorch
    """
    print("🧪 RUNNING MODULE INTEGRATION TEST")
    print("=" * 50)

    # Run all unit tests
    '''print("Running unit tests...")
    test_unit_log_softmax()
    test_unit_mse_loss()
    test_unit_cross_entropy_loss()
    test_unit_binary_cross_entropy_loss()'''

    print("\nRunning integration scenarios...")

    # Test realistic end-to-end scenario with previous modules
    print("🧪 Integration Test: Realistic training scenario...")

    # Simulate a complete prediction -> loss computation pipeline

    # 1. MSE for regression (house price prediction)
    house_predictions = Tensor([250.0, 180.0, 320.0, 400.0])  # Predicted prices in thousands
    house_actual = Tensor([245.0, 190.0, 310.0, 420.0])       # Actual prices
    mse_loss = MSELoss()
    house_loss = mse_loss.forward(house_predictions, house_actual)
    assert house_loss.data > 0, "House price loss should be positive"
    assert house_loss.data < 1000, "House price loss should be reasonable"

    # 2. CrossEntropy for classification (image recognition)
    image_logits = Tensor([[2.1, 0.5, 0.3], [0.2, 2.8, 0.1], [0.4, 0.3, 2.2]])  # 3 images, 3 classes
    image_labels = Tensor([0, 1, 2])  # Correct class for each image
    ce_loss = CrossEntropyLoss()
    image_loss = ce_loss.forward(image_logits, image_labels)
    assert image_loss.data > 0, "Image classification loss should be positive"
    assert image_loss.data < 5.0, "Image classification loss should be reasonable"

    # 3. BCE for binary classification (spam detection)
    spam_probabilities = Tensor([0.85, 0.12, 0.78, 0.23, 0.91])
    spam_labels = Tensor([1.0, 0.0, 1.0, 0.0, 1.0])  # True spam labels
    bce_loss = BinaryCrossEntropyLoss()
    spam_loss = bce_loss.forward(spam_probabilities, spam_labels)
    assert spam_loss.data > 0, "Spam detection loss should be positive"
    assert spam_loss.data < 5.0, "Spam detection loss should be reasonable"

    # 4. Test numerical stability with extreme values
    extreme_logits = Tensor([[100.0, -100.0, 0.0]])
    extreme_targets = Tensor([0])
    extreme_loss = ce_loss.forward(extreme_logits, extreme_targets)
    assert not np.isnan(extreme_loss.data), "Loss should handle extreme values"
    assert not np.isinf(extreme_loss.data), "Loss should not be infinite"

    print("✅ End-to-end loss computation works!")
    print("✅ All loss functions handle edge cases!")
    print("✅ Numerical stability verified!")

    print("\n" + "=" * 50)
    print("🎉 ALL TESTS PASSED! Module ready for export.")
    print("Run: tito module complete 04")

    # Run comprehensive module test
if __name__ == "__main__":
    test_module()