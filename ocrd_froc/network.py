import os
import copy
import numpy as np
import torch
import torch.nn.functional as F
import json

class Backbone(torch.nn.Module):
    def __init__(self, output_dim=64):
        super().__init__()
        self.act = torch.nn.LeakyReLU()
        self.max_pool1 = torch.nn.MaxPool2d(kernel_size=(4,2))
        self.max_pool2 = torch.nn.MaxPool2d(kernel_size=(1,2))
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=8, kernel_size= (3, 2), stride=(2,1))
        self.conv2 = torch.nn.Conv2d(in_channels=8, out_channels=32, kernel_size=(6, 4), padding=(3,1))
        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=output_dim, kernel_size=(3, 3), padding=(1,1))
        self.padding_params = 1/8
        self.output_dim = output_dim

    def forward(self, x):
        x = torch.nn.functional.pad(x,(1,2,1,2))
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.max_pool1(x)
        x = self.act(self.conv3(x))
        x = self.max_pool2(x)
        return x

class Backbone2L(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.act = torch.nn.ReLU()
        self.max_pool = torch.nn.MaxPool2d(kernel_size=(2,2), stride=(2,1))
        self.conv1 = torch.nn.Conv2d(1,  64,  (3, 3), stride=(3,2), padding=(2,2))
        self.conv2 = torch.nn.Conv2d(64, 128, (3, 3), stride=(3,2), padding=(2,2))
        self.output_dim = 128

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.max_pool(x)
        x = self.act(self.conv2(x))
        x = self.max_pool(x)
        return x

class NoDimRedBackbone(torch.nn.Module):
    def __init__(self, output_dim=32):
        super().__init__()
        self.act = torch.nn.LeakyReLU()
        self.max_pool2 = torch.nn.MaxPool2d(kernel_size=(3,1))
        self.conv1 = torch.nn.Conv2d(1, 8, (7, 5), stride=(1,1), padding='same')
        self.conv2 = torch.nn.Conv2d(8, 32, (7, 5), padding='same')
        self.conv3 = torch.nn.Conv2d(32, output_dim, (3, 3), padding='same')
        self.padding_params = 1/8
        self.output_dim = output_dim

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.max_pool2(x)
        x = self.act(self.conv3(x))
        x = self.max_pool2(x)
        return x.mean(axis=2)

class OCROnly(torch.nn.Module):
    # Here I changed lstm_layers from 3 to 1 for the bug hunt, and Backbone to Backbone2L
    def __init__(self, nb_classes, feature_dim=200, backbone=Backbone, lstm_layers=3):
        super().__init__()
        self.backbone = backbone()
        self.embed = torch.nn.Linear(self.backbone.output_dim, feature_dim)
        self.rnn  = torch.nn.LSTM(feature_dim, feature_dim, lstm_layers, bidirectional=True)
        self.head = torch.nn.Linear(2*feature_dim, nb_classes)
        self.act = torch.nn.ReLU()
        self.__length_map = {}
        self.__init_length_map()
    
    def save(self, folder):
        os.makedirs(folder, exist_ok=True)
        torch.save(self.backbone.state_dict(), os.path.join(folder, 'backbone.pth'))
        torch.save(self.embed.state_dict(), os.path.join(folder, 'embed.pth'))
        torch.save(self.rnn.state_dict(), os.path.join(folder, 'rnn.pth'))
        torch.save(self.head.state_dict(), os.path.join(folder, 'head.pth'))
    
    def load(self, folder, device='cuda:0'):
        self.backbone.load_state_dict(torch.load(os.path.join(folder, 'backbone.pth'), map_location=device))
        self.embed.load_state_dict(torch.load(os.path.join(folder, 'embed.pth'), map_location=device))
        self.rnn.load_state_dict(torch.load(os.path.join(folder, 'rnn.pth'), map_location=device))
        self.head.load_state_dict(torch.load(os.path.join(folder, 'head.pth'), map_location=device))
    
    def convert_widths(self, w, max_width):
        return torch.Tensor([min(self.__length_map[x], max_width) for x in w]).long()
    
    def __init_length_map(self):
        max_length = 2000
        try:
            self.__length_map = json.load(open(os.path.join('models', 'length_map.json'), 'rt'))
            return
        except: pass
        
        tns = torch.zeros(1, 1, 8, max_length)
        with torch.no_grad():
            out  = self.backbone(tns)
            last = out[0][0][0][out.shape[3]//2]
            ls  = 0
            pos = 0
            self.__length_map = []
            for i in range(max_length):
                tns[0,0,:,i] = i
                out = torch.sum(self.backbone(tns), axis=1)
                while pos<out.shape[2]-1 and out[0,0,pos]!=out[0,0,pos+1]:
                    pos += 1
                self.__length_map.append(pos-1)
        json.dump(self.__length_map, open(os.path.join('models', 'length_map.json'), 'wt'))
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.act(x)
        # ~ x, _ = torch.max(x, axis=2)
        x = torch.mean(x, axis=2)
        x = x.permute(2, 0, 1)
        x = self.embed(x)
        x, _ = self.rnn(x)
        x = self.head(x)
        return x

class ColClassifier(torch.nn.Module):
    def __init__(self, backbone, feature_dim, nb_classes=12):
        super().__init__()
        self.backbone = backbone
        self.embed = torch.nn.Linear(self.backbone.output_dim, feature_dim)
        self.rnn  = torch.nn.LSTM(feature_dim, feature_dim, 2, bidirectional=True)
        self.head = torch.nn.Linear(2*feature_dim, nb_classes)
    
    def forward(self, x):
        x = self.embed(self.backbone(x).transpose(1,2).transpose(0,1))
        x, _ = self.rnn(x)
        x = self.head(x)
        return x.transpose(0,1)
    
    def save(self, folder):
        os.makedirs(folder, exist_ok=True)
        torch.save(self.backbone.state_dict(), os.path.join(folder, 'backbone.pth'))
        torch.save(self.embed.state_dict(), os.path.join(folder, 'embed.pth'))
        torch.save(self.rnn.state_dict(), os.path.join(folder, 'rnn.pth'))
        torch.save(self.head.state_dict(), os.path.join(folder, 'head.pth'))
    
    def load(self, folder, device='cuda:0'):
        self.backbone.load_state_dict(torch.load(os.path.join(folder, 'backbone.pth'), map_location=device))
        self.embed.load_state_dict(torch.load(os.path.join(folder, 'embed.pth'), map_location=device))
        self.rnn.load_state_dict(torch.load(os.path.join(folder, 'rnn.pth'), map_location=device))
        self.head.load_state_dict(torch.load(os.path.join(folder, 'head.pth'), map_location=device))

class SelectiveOCR(torch.nn.Module):
    def __init__(self, classifier, models):
        super().__init__()
        self.classifier = classifier
        self.models     = models
    
    def forward(self, x, model_idx=None):
        if x.shape[0]!=1:
            raise Exception('SelectiveOCR cannot work on batches, sorry')
        if model_idx == None:
            scores = self.classifier(x).sum(axis=1)#.view(-1,13).mean(axis=1)
            n = torch.argmax(scores[0,:]).item()
            model_idx = n
        model_idx = model_idx if model_idx in self.models else 0
        return self.models[model_idx](x)

class SplitOCR(torch.nn.Module):
    def __init__(self, classifier, models):
        super().__init__()
        self.classifier = classifier
        self.models     = models
    
    def forward(self, x, labels=None):
        if x.shape[0]!=1:
            raise Exception('SplitOCR cannot work on batches, sorry')
        self.classifier.eval()
        for m in self.models:
            self.models[m].eval()
        if labels is None:
            scores = self.classifier(x)[0, :, :]
            n = torch.argmax(scores, axis=1)
        else:
            n = labels
        seq = []
        a = 0
        while a<n.shape[0]-1:
            for b in range(a+1, n.shape[0]):
                if n[b]!=n[a]:
                    break
            seq.append(((a,b), n[a].item()))
            a = b
        res = []
        for s, n in seq:
            if s[1]-s[0]<2:
                continue
            n = n if n in self.models else 0
            r = self.models[n](x[:, :, :, s[0]:s[1]])
            res.append(r)
        return torch.concat(res, axis=0)

class COCR(torch.nn.Module):
    def __init__(self, classifier, models):
        super().__init__()
        self.classifier = classifier
        self.models     = models
    
    def convert_widths(self, w, max_width):
        return self.models[0].convert_widths(w, max_width)
    
    def save(self, folder):
        self.classifier.save(os.path.join(folder, 'classifier'))
        for n in self.models:
            self.models[n].save(os.path.join(folder, '%s' % n))
    
    def load(self, folder, device='cuda:0'):
        self.classifier.load(os.path.join(folder, 'classifier'), device=device)
        for n in self.models:
            self.models[n].load(os.path.join(folder, '%s' % n), device=device)
        
    def forward(self, x, fast_cocr=True):
        scores = F.softmax(self.classifier(x), dim=2)
        res = 0
        for n in self.models:
            s = scores[:, :, n].unsqueeze(-1)
            if fast_cocr and torch.max(s)<0.1:
                continue
            y = self.models[n](x)
            res += y * s
        return res

# Same CNN/embedding, different RNN
class COCR2(torch.nn.Module):
    def __init__(self, classifier, generic_model, nb_models=13):
        super().__init__()
        self.classifier = classifier
        self.generic = generic_model
        self.cnn = generic_model.backbone
        self.embed = generic_model.embed
        self.rnn = {
            x: copy.deepcopy(generic_model.rnn) for x in range(nb_models)
        }
        self.head = {
            x: copy.deepcopy(generic_model.head) for x in range(nb_models)
        }
        for n in self.rnn:
            self.rnn[n].load_state_dict(self.rnn[n].state_dict()) # not sure, maybe needed
            self.rnn[n].flatten_parameters()
        
    
    def convert_widths(self, w, max_width):
        return self.generic.convert_widths(w, max_width)
    
    def forward(self, x):
        scores = F.softmax(self.classifier(x), dim=2)
        x = self.cnn(x)
        x = self.generic.act(x)
        x, _ = torch.max(x, axis=2)
        x = x.permute(2, 0, 1)
        x = self.embed(x)
        
        res = 0
        for n in self.rnn:
            y, _ = self.rnn[n](x)
            y = self.head[n](y)
            res += y * scores[:, :, n].unsqueeze(-1)
        return res
