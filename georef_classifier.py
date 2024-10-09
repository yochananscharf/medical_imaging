import torch
import torchvision
from torchvision import transforms
import os
import pickle
import json
from PIL import Image
import numpy as np
from sklearn.svm import SVC, SVR
from timeit import default_timer as timer

MODEL_SVM_NAME = 'model_svm_new'

USE_GPU = False
# if torch.cuda.is_available():
#     USE_GPU = True




class GeorefClassifier:
    def __init__(self) -> None:
        self.model_svm = None
        self.model_embeddings = None
        self.transform = None
        self.load_model_embeddings()
        self.load_model_svm()
        pass

    def load_model_svm(self, model_name=MODEL_SVM_NAME):
        model_path = model_name+'.pkl'
        with open(model_path, 'rb') as f:
            clf = pickle.load(f)
        # clf = SVC(kernel='rbf', C=0.4)
        self.model_svm = clf
        return 

    def load_model_embeddings(self):
        #self.model_embeddings = torchvision.models.swin_transformer.swin_v2_b()
        self.model_embeddings = torchvision.models.swin_transformer.swin_v2_t()
        #full_state_dict = torch.load('satlas-model-v1-highres.pth',map_location=torch.device('cpu'))
        full_state_dict = torch.load('sentinel2_swint_si_rgb.pth',map_location=torch.device('cpu'))

        
        swin_prefix = 'backbone.backbone.'
        swin_state_dict = {k[len(swin_prefix):]: v for k, v in full_state_dict.items() if k.startswith(swin_prefix)}
        self.model_embeddings.load_state_dict(swin_state_dict)
        transform = transforms.Compose([
            transforms.ToTensor(),
            # pth_transforms.Normalize((123.675, 116.28, 103.53), (58.395, 57.12, 57.375)),
        ])
        if USE_GPU:
            self.model_embeddings.cuda()
        self.model_embeddings.eval()


        
        self.transform = transform
        return 


    def predict_image_vec(self, image_path):
        image_pil = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(image_pil)
        img_tensor = img_tensor.unsqueeze(0)
        with torch.no_grad():
            embeddings_vec = self.model_embeddings(img_tensor)
        prediction_vec_arr = embeddings_vec.numpy()
        return prediction_vec_arr

    def predict_image(self, image_pil):
        
        img_tensor = self.transform(image_pil)
        img_tensor = img_tensor.unsqueeze(0)
        with torch.no_grad():
            embeddings_vec = self.model_embeddings(img_tensor)
        prediction_vec_arr = embeddings_vec.numpy()
        y_predict = self.model_svm.predict(prediction_vec_arr)
        return y_predict
    

    


georef_classifier = GeorefClassifier()

def read_image(path: str) -> str:
    image_pil = Image.open(path).convert('RGB')
    return image_pil

def predict_images_vecs(images_paths):
    images_pil = [read_image(img_path) for img_path in images_paths]
    prediction_list  = []
    for img in images_pil:
        img_tensor = georef_classifier.transform(img)
        img_tensor = img_tensor.unsqueeze(0)
        with torch.no_grad():
            embeddings_vec = georef_classifier.model_embeddings(img_tensor)
        prediction_vec_arr = embeddings_vec.numpy()
        prediction_list.append(prediction_vec_arr.tolist())
    
    return prediction_list

def predict_images_vecs_batch(images_paths):
    batch_size = 128
    num_images = len(images_paths)
    count = 0
    prediction_list  = []
    
    while count < num_images:
        #start = timer() 
        if num_images - count < batch_size:
            batch_size = num_images - count
            
        images_pil = [read_image(img_path) for img_path in images_paths[count:count+batch_size]]

        img_tensor_list = []
        for img in images_pil:
            img_tensor = georef_classifier.transform(img)
            #imgs_tensor = img_tensor.unsqueeze(0)
            if USE_GPU:
                img_tensor = img_tensor.cuda()
            img_tensor_list.append(img_tensor)
        imgs_tensor = torch.stack(img_tensor_list)

        with torch.no_grad():
            embeddings_vec = georef_classifier.model_embeddings(imgs_tensor)
        if USE_GPU:
            embeddings_vec = embeddings_vec.cpu()
        prediction_vec_arr = embeddings_vec.numpy().tolist()
        prediction_list.extend(prediction_vec_arr)
        count+=batch_size
    return prediction_list


def predict_class(embeddings):
    embeddings = np.array(embeddings)
    embeddings = embeddings.reshape((-1,1000))
    georef_classifier.load_model_svm()
    y_predict = georef_classifier.model_svm.predict(embeddings)
    return y_predict


def prepare_training_data(training_dict):
    image_paths = list(training_dict.keys())
    labels = list(training_dict.values())
    return image_paths, labels

def extract_embeddings(images):
    vecs_list = []
    for img in images:
        vec = georef_classifier.predict_image_vec(img)
        vecs_list.append(vec)
    return vecs_list

def train_classifier(images_paths, labels, classifier_name='model_svm_new.pkl'):
    #embeddings = extract_embeddings(images_paths)
    embeddings = predict_images_vecs_batch(images_paths)
    embeddings = np.array(embeddings)
    embeddings = embeddings.reshape((-1,1000))
    #georef_classifier.model_svm = SVC(kernel='rbf', C=0.4)
    georef_classifier.model_svm = SVR(kernel='rbf', C=0.4)
    georef_classifier.model_svm.fit(embeddings, labels)
    with open(classifier_name,'wb') as f:
        pickle.dump(georef_classifier.model_svm,f)
    #print metrics


def create_train_json(dir_pos, dir_neg, json_name='data_train.json'):
    images_pos = list(os.listdir(dir_pos))
    images_neg = list(os.listdir(dir_neg))
    images_pos = [dir_pos+'/' + img_p for img_p in images_pos]
    images_neg = [dir_neg+'/'  + img_n for img_n in images_neg]
    labels_pos = [1]*len(images_pos)
    labels_neg = [0]*len(images_neg)
    images_all = images_pos+images_neg
    labesl_all = labels_pos+labels_neg
    data_dict = dict(zip(images_all, labesl_all))

    with open(json_name, "w") as outfile:
        json.dump(data_dict, outfile)