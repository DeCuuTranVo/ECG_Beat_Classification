from src.predicter import Predicter


def predict():
    """
    Predict image in sample_dir with trained model

    Args:
        sample_dir (str): Directory of image file to be predicted.

    Returns:
        prediction (dict): Dictionary of propability of 2 classes,
            and predicted class of the image.
    """

    # Create predictor object
    # os.chdir('./src_Tran')
    predicter = Predicter()
    
    # Predict the image in sample directory
    # os.chdir('..')
    prediction = predicter.predict()
    return prediction


if __name__ == '__main__':
    prediction = predict()
    print(prediction)