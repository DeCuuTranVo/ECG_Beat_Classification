from src.model import NeuralNetwork
import torch
from torchvision import transforms
import json
from src.trainer import CustomTrainer
import os

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import scipy.io


class Predicter():
    '''
    Predict the label of new image base on trained model
    '''

    def __init__(self, model_type='mitbih', using_gpu=True):
        """
        Construct the predicter object.

        Args:
            model_type (str, optional): Type of model architecture.
            using_gpu (bool, optional): GPU enable option. Defaults to True.
        """

        # Load training parameters
        params = json.load(open('config/config.json', 'r'))

        # Create CustomTrainer instance with loaded training parameters
        trainer = CustomTrainer(**params)

        # Check device
        self.device = 'cuda' if torch.cuda.is_available() and using_gpu else 'cpu'

        # Create model
        self.model = NeuralNetwork(trainer.ARCHITECTURE, trainer.INPUT_DIMENSION, trainer.DATASET).to(self.device)

        # Load trained model
        self.model.load_state_dict(torch.load(os.path.join(
            trainer.MODEL_DIR, "trial-" + trainer.TRIAL + ".pth")))

        # Switch model to evaluation mode
        self.model.eval()

        # # Image processing
        # self.height = 224
        # self.width = self.height * 1
        # self.transform = transforms.Compose([
        #     transforms.Resize(
        #         (int(self.width),
        #         int(self.height))),
        #     transforms.ToTensor(),
        #     transforms.Normalize(
        #         [0.485, 0.456, 0.406],
        #         [0.229, 0.224, 0.225])])

        self.signal_path = trainer.PREDICT_SIGNAL_PATH
        # print(self.signal_path)


    def predict(self):
        """
        Predict image in image_path is peripheral or central.

        Args:
            image_path (str): Directory of image file.

        Returns:
            result (dict): Dictionary of propability of 2 classes,
                and predicted class of the image.
        """
        # Readin .dat file
        mat = scipy.io.loadmat(self.signal_path)
        signal = mat['data'][0]

        signal = torch.from_numpy(signal)
        signal = signal.float()
        # print(signal.shape)
        # exit()
        # signal = self.transform(signal)
        signal = signal.view(1, 1, *signal.size()).to(self.device)
        
        # Result
        labels = ["N", "S", "P", "F", "U"]
        
        result = {'prob_N': 0, 'prob_S': 0, 'prob_P': 0, 'prob_F': 0, 'prob_U': 0, 'label': ''}

        # Predict image
        with torch.no_grad():
            output = self.model(signal)
            ps = torch.nn.functional.softmax(output, dim=1)
            # ps = torch.exp(output)
            # print(ps)
            
            result['prob_N'] = float(ps[0][0].item())
            result['prob_S'] = float(ps[0][1].item())
            result['prob_P'] = float(ps[0][2].item())
            result['prob_F'] = float(ps[0][3].item())
            result['prob_U'] = float(ps[0][4].item())
            
            label_index = torch.argmax(ps).item()
            result['label'] = labels[label_index]
        return result


if __name__ == "__main__":
    my_predicter = Predicter()