from train import train_model
from inference import predict_price

def test_training_creates_model():
    assert train_model().exists()

def test_prediction_is_positive_number():
    train_model()
    result = predict_price(800)
    assert isinstance(result, float)
    assert result > 0
