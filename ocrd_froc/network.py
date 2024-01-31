import os
import torch    # type: ignore
import torch.nn.functional as F    # type: ignore
import json
from os.path import join

def load_network(folder):
    """
    Loads a network stored in a folder. The network should be among the
    ones below, other it will fail for sure.

    :param folder: folder containing the files
    :return: a network which can be used for OCR
    """
    p = json.load(open(join(folder, 'params.json'), 'rt'))
    if p['type']=='OCROnly':
        return OCROnly.load_from_folder(folder)
    if p['type']=='COCR':
        return COCR.load_from_folder(folder)
    if p['type']=='SelOCR':
        return SelOCR.load_from_folder(folder)
    if p['type']=='SimpleHead':
        return SimpleHead.load_from_folder(folder)
    if p['type']=='DHCOCR':
        return DHCOCR.load_from_folder(folder)
    raise Exception('Unknown network type: %s' % p['type'])

class Backbone(torch.nn.Module):
    """
    CNN used in all of our OCR pipelines.
    """
    def __init__(self, output_dim=64):
        """
        Constructor

        :param output_dim: number of neurons in the output layer
        :return: instance of the class
        """
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
        """
        Extracts features from an input text line

        :param x: text line (batch)
        :return: descriptors (batch)
        """
        x = torch.nn.functional.pad(x,(1,2,1,2))
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.max_pool1(x)
        x = self.act(self.conv3(x))
        x = self.max_pool2(x)
        return x

class NoDimRedBackbone(torch.nn.Module):
    """
    CNN which does not reduce horizontal dimensions of the input.
    Useful for pixel column labeling.
    """
    def __init__(self, output_dim=32):
        """
        Constructor

        :param output_dim: number of neurons in the output layer
        :return: instance of the class
        """
        super().__init__()
        self.act = torch.nn.LeakyReLU()
        self.max_pool2 = torch.nn.MaxPool2d(kernel_size=(3,1))
        self.conv1 = torch.nn.Conv2d(1, 8, (7, 5), stride=(1,1), padding='same')
        self.conv2 = torch.nn.Conv2d(8, 32, (7, 5), padding='same')
        self.conv3 = torch.nn.Conv2d(32, output_dim, (3, 3), padding='same')
        self.padding_params = 1/8
        self.output_dim = output_dim

    def forward(self, x):
        """
        Extracts features from an input text line

        :param x: text line (batch)
        :return: descriptors (batch)
        """
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.max_pool2(x)
        x = self.act(self.conv3(x))
        x = self.max_pool2(x)
        return x.mean(axis=2)

class SimpleHead(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.rnn = torch.nn.LSTM(input_dim, output_dim, 1, bidirectional=True)
        self.lin = torch.nn.Linear(2*output_dim, output_dim)
        self.params = {'type': 'SimpleHead', 'input_dim': input_dim, 'output_dim': output_dim}

    def load_from_folder(folder):
        """
        Static method loading an instance from a folder

        :param folder: source folder
        :return: an instance of the class
        """
        p = json.load(open(join(folder, 'params.json'), 'rt'))
        net = SimpleHead(input_dim=p['input_dim'], output_dim=p['output_dim'])
        net.load(folder)
        return net

    def load(self, folder):
        """
        Loads the models' weights from a folder. Note that the model has
        to be properly initialized first.

        :param folder: source folder
        """
        self.load_state_dict(torch.load(join(folder, 'net.pth')))
        self.params = json.load(open(join(folder, 'params.json'), 'rt'))

    def save(self, folder):
        """
        Saves the model to a folder. Creates this folder if needed.

        :param folder: destination folder
        """
        os.makedirs(folder, exist_ok=True)
        torch.save(self.state_dict(), join(folder, 'net.pth'))
        with open(join(folder, 'params.json'), 'wt') as f:
            json.dump(self.params, f, indent=4)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.lin(x)
        return x

class OCROnly(torch.nn.Module):
    """
    This is the base of all OCR models used in this project. It combines
    a CNN, an embedding layer, one or several LSTM layers, and an output
    layer.
    """
    def __init__(self, nb_classes, feature_dim=200, backbone=Backbone, lstm_layers=3):
        """
        Constructor of the class

        :param nb_classes: number of characters composing the alphabet
        :param feature_dim: number of neurons in the LSTM layers
        :param backbone: class of the backbone
        :param lstm_layers: self-describing parameter
        :return: an instance of the class
        """
        super().__init__()
        self.params = {'type': 'OCROnly', 'nb_classes': nb_classes, 'feature_dim': feature_dim, 'lstm_layers': lstm_layers}
        self.backbone = backbone()
        self.embed = torch.nn.Linear(self.backbone.output_dim, feature_dim)
        self.rnn  = torch.nn.LSTM(feature_dim, feature_dim, lstm_layers, bidirectional=True)
        self.head = torch.nn.Linear(2*feature_dim, nb_classes)
        self.act = torch.nn.ReLU()
        self.__length_map = {}
        self.__init_length_map()

    def load_from_folder(folder):
        """
        Static method loading an instance from a folder

        :param folder: source folder
        :return: an instance of the class
        """
        p = json.load(open(join(folder, 'params.json'), 'rt'))
        net = OCROnly(nb_classes=p['nb_classes'], feature_dim=p['feature_dim'], lstm_layers=p['lstm_layers'])
        net.load(folder)
        return net

    def save(self, folder):
        """
        Saves the model to a folder. Creates this folder if needed.

        :param folder: destination folder
        """
        os.makedirs(folder, exist_ok=True)
        torch.save(self.backbone.state_dict(), join(folder, 'backbone.pth'))
        torch.save(self.embed.state_dict(), join(folder, 'embed.pth'))
        torch.save(self.rnn.state_dict(), join(folder, 'rnn.pth'))
        torch.save(self.head.state_dict(), join(folder, 'head.pth'))
        with open(join(folder, 'params.json'), 'wt') as f:
            json.dump(self.params, f, indent=4)

    def load(self, folder):
        """
        Loads the models' weights from a folder. Note that the model has
        to be properly initialized first.

        :param folder: source folder
        """
        self.backbone.load_state_dict(torch.load(join(folder, 'backbone.pth')))
        self.embed.load_state_dict(torch.load(join(folder, 'embed.pth')))
        self.rnn.load_state_dict(torch.load(join(folder, 'rnn.pth')))
        self.head.load_state_dict(torch.load(join(folder, 'head.pth')))
        self.params = json.load(open(join(folder, 'params.json'), 'rt'))

    def get_optimizers(self, folder=None):
        """
        Returns an array containing one optimizer - if folder is not none,
        then the optimizer's state dict stored in the folder is loaded.

        :param folder: source folder
        :return: an array containing one or optimizer
        """
        res = [
            torch.optim.Adam(self.parameters(), lr=0.001)
        ]
        if folder is not None:
            res[0].load_state_dict(torch.load(join(folder, 'optimizer.pth')))
        return res

    def save_optimizers(self, optimizers, folder):
        """
        Save the optimizer in a given folder.

        :param folder: destination folder
        """
        torch.save(optimizers[0].state_dict(), join(folder, 'optimizer.pth'))

    def convert_widths(self, w, max_width):
        """
        Converts an input widths (in pixel columns) to output widths (in
        the output tensor). Returned as tensor.

        :param w: tensor or array containing a list of width
        :return: long tensor containing the converted widths
        """
        return torch.Tensor([min(self.__length_map[x], max_width) for x in w]).long()

    def __init_length_map(self):
        """
        Initializes the map conversion system for convert_width(). Note
        that it tries to cache the resuts in dat/length_map.json.
        """
        max_length = 2000
        try:
            with open(join('dat', 'length_map.json'), 'rt', encoding='utf-8') as f:
                self.__length_map = json.load(f)
                return
        except:
            # Which exceptions are caught here and why?
            pass

        tns = torch.zeros(1, 1, 8, max_length)
        with torch.no_grad():
            out  = self.backbone(tns)
            pos = 0
            self.__length_map = []
            for i in range(max_length):
                tns[0,0,:,i] = i
                out = torch.sum(self.backbone(tns), axis=1)
                while pos<out.shape[2]-1 and out[0,0,pos]!=out[0,0,pos+1]:
                    pos += 1
                self.__length_map.append(pos-1)
        with open(join('dat', 'length_map.json'), 'wt', encoding='utf-8'):
            json.dump(self.__length_map, f)

    def forward(self, x):
        """
        Processes an input batch.

        :param x: input batch
        :return: the network's output, ready to be convered to a string
        """
        x = self.backbone(x)
        x = self.act(x)
        x = torch.mean(x, axis=2)
        x = x.permute(2, 0, 1)
        x = self.embed(x)
        x, _ = self.rnn(x)
        x = self.head(x)
        return x

class ColClassifier(torch.nn.Module):
    """
    This class is used for classifying the pixel columns of a text line.
    It is getting outdated and might be removed at a later stage.
    """
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
        torch.save(self.backbone.state_dict(), join(folder, 'backbone.pth'))
        torch.save(self.embed.state_dict(), join(folder, 'embed.pth'))
        torch.save(self.rnn.state_dict(), join(folder, 'rnn.pth'))
        torch.save(self.head.state_dict(), join(folder, 'head.pth'))

    def load(self, folder):
        self.backbone.load_state_dict(torch.load(join(folder, 'backbone.pth')))
        self.embed.load_state_dict(torch.load(join(folder, 'embed.pth')))
        self.rnn.load_state_dict(torch.load(join(folder, 'rnn.pth')))
        self.head.load_state_dict(torch.load(join(folder, 'head.pth')))

class SelOCR(torch.nn.Module):
    """
    SelOCR stands for Selective OCR, not Selachophile OCR. This class
    groups together a font group classifier and several OCR models. When
    processing a text line, it first classifies it, and then applies the
    best OCR model for the detected font group.

    If there is no OCR model for this font group, then model number 0
    is used, which implies:
    a) there must be a model number 0,
    b) model number 0 should be a generic model, not a specialized one
    """
    def __init__(self, classifier, models):
        """
        Constructor of the class. Make sure to include a model with
        key 0 in the model map, otherwise another one will be used
        instead and this can lead to poor behavior.

        :param classifier: a classifier which output can be averaged on axis 1 to make a global line prediction
        :param models: a map from integer class number (accordingly to the classifier) to OCR models specialized for the corresponding font groups
        :return: an instance of the class
        """
        super().__init__()
        self.classifier = classifier
        self.models     = models # 0 has to be the baseline
        self.params = {'type': 'SelOCR'}
        if self.models is not None and not 0 in self.models:
            self.models[0] = self.models[min(self.models)]

    def save(self, folder):
        """
        Saves the classifier and OCR models to a folder.
        Creates this folder if needed.

        :param folder: destination folder
        """
        self.classifier.save(join(folder, 'classifier'))
        for n in self.models:
            self.models[n].save(join(folder, '%s' % n))
        self.params['models'] = self.models.keys()
        with open(join(folder, 'params.json'), 'wt') as f:
            json.dump(self.params, f, indent=4)

    def load(self, folder):
        """
        Loads the classifier and OCR models from a folder.

        :param folder: source folder
        """
        self.params = json.load(open(join(folder, 'params.json'), 'rt', encoding='utf-8'))
        self.classifier = load_network(join(folder, 'classifier'))
        self.models = {n: load_network(join(folder, '%d' % n)) for n in self.params['models']}
        if not 0 in self.models:
            self.models[0] = self.models[min(self.models)]

    def load_from_folder(folder):
        """
        Static method. Loads an instance of the class from a specific
        folder.

        :param folder: source folder
        :return: a SelOCR instance
        """
        net = SelOCR(None, None)
        net.load(folder)
        return net

    def to(self, device):
        """
        Overriding Pytorch's to() method, as it doesn't cope well
        with maps.

        :param device: device to send the model to
        :return: the instance on the new device
        """
        self.classifier = self.classifier.to(device)
        for m in self.models:
            self.models[m] = self.models[m].to(device)
        return self

    def forward(self, x, model_idx=None):
        """
        Processes an input batch - batch which must contain only one
        single text line (because of the branching).

        :param x: input batch
        :return: the network's output, ready to be convered to a string
        """
        if x.shape[0] != 1:
            raise ValueError('SelOCR cannot work on batches containing multiple inputs, sorry')
        if model_idx == None:
            scores = self.classifier(x).sum(axis=1)#.view(-1,13).mean(axis=1)
            n = torch.argmax(scores[0,:]).item()
            model_idx = n
        model_idx = model_idx if model_idx in self.models else 0
        return self.models[model_idx](x)

class COCR(torch.nn.Module):
    """
    COCR stands for Combined OCR. This system combines multiple OCR
    models, weighting their outputs locally based on font group classification
    scores. It is slower than the SelOCR, but performs better on text
    lines containing multiple font groups.
    """
    def __init__(self, classifier, models):
        """
        Constructor of the class. Provide it with a classifier, and a
        map of models. It is recommended to actually use a baseline
        model (multiple instances of it) and fine-tune it.

        :param classifier: a classifier which output has the same length as OCR outputs. You can use an OCR model there.
        :param models: a map from integer class number (accordingly to the classifier) to OCR models
        :return: an instance of the class
        """
        super().__init__()
        self.classifier = classifier
        self.models = models
        self.params = {'type': 'COCR'}

    def convert_widths(self, w, max_width):
        """
        Converts an input widths (in pixel columns) to output widths (in
        the output tensor). Returned as tensor.

        :param w: tensor or array containing a list of width
        :return: long tensor containing the converted widths
        """
        return self.models[self.params['models'][0]].convert_widths(w, max_width)

    def save(self, folder):
        """
        Saves the classifier and OCR models to a folder.
        Creates this folder if needed.

        :param folder: destination folder
        """
        self.classifier.save(join(folder, 'classifier'))
        for n in self.models:
            self.models[n].save(join(folder, '%s' % n))
        self.params['models'] = [x for x in self.models]
        with open(join(folder, 'params.json'), 'wt', encoding='utf-8') as f:
            json.dump(self.params, f, indent=4)

    def to(self, device):
        """
        Overriding Pytorch's to() method, as it doesn't cope well
        with maps.

        :param device: device to send the model to
        :return: the instance on the new device
        """
        self.classifier = self.classifier.to(device)
        for m in self.models:
            self.models[m] = self.models[m].to(device)
        return self

    def parameters(self):
        """
        Overriding Pytorch's parameters() method, as it doesn't cope well
        with maps.

        :return: all trainable parameters of the COCR
        """
        res = [x for x in self.classifier.parameters()]
        for n in self.models:
            res += [x for x in self.models[n].parameters()]
        return res

    def load_from_folder(folder):
        """
        Static method. Loads an instance of the class from a specific
        folder.

        :param folder: source folder
        :return: a COCR
        """
        net = COCR(None, None)
        net.load(folder)
        return net

    def load(self, folder):
        """
        Loads the classifier and OCR models from a folder.

        :param folder: source folder
        """
        with open(join(folder, 'params.json'), 'rt') as f:
            self.params = json.load(f)
        self.classifier = load_network(join(folder, 'classifier'))
        self.models = {n: load_network(join(folder, '%d' % n)) for n in self.params['models']}

    def get_optimizers(self, folder=None):
        """
        Returns an array containing optimizers for each part of the
        COCR - if folder is not none, then the optimizers' state dict
        stored in the folder are loaded.

        :param folder: source folder
        :return: an array containing several optimizers
        """
        res = [
            torch.optim.Adam(self.classifier.parameters(), lr=0.001)
        ] + [
            torch.optim.Adam(self.models[n].parameters(), lr=0.001) for n in sorted(self.models)
        ]
        if folder is not None:
            res[0].load_state_dict(torch.load(join(folder, 'classifier', 'optimizer.pth')))
            for i, n in enumerate(sorted(self.models)):
                res[i+1].load_state_dict(torch.load(join(folder, '%d' % n, 'optimizer.pth')))
        return res

    def save_optimizers(self, optimizers, folder):
        """
        Save the optimizer in a given folder.

        :param folder: destination folder
        """
        torch.save(optimizers[0].state_dict(), join(folder, 'classifier', 'optimizer.pth'))
        for i, n in enumerate(sorted(self.models)):
            torch.save(optimizers[i+1].state_dict(), join(folder, '%d' % n, 'optimizer.pth'))

    def forward(self, x, fast_cocr=True):
        """
        Processes an input batch

        :param x: input batch
        :return: the network's output, ready to be convered to a string
        """
        scores = F.softmax(self.classifier(x), dim=2)
        res = 0
        for n in self.models:
            s = scores[:, :, n].unsqueeze(-1)
            if fast_cocr and torch.max(s) < 0.1:
                continue
            y = self.models[n](x)
            res += y * s
        return res

class DHCOCR(torch.nn.Module):
    """
    DHCOCR stands for Double-Headed Combined OCR. This system combines multiple OCR
    models, weighting their outputs locally based on font group classification
    scores, and has two heads - one for OCR, one for character-level font recognition.
    """
    def __init__(self, classifier, models):
        """
        Constructor of the class. Provide it with a classifier, and a
        map of models. It is recommended to actually use a baseline
        model (multiple instances of it) and fine-tune it.

        :param classifier: a classifier which output has the same length as OCR outputs. You can use an OCR model there.
        :param models: a map from integer class number (accordingly to the classifier) to OCR models
        :return: an instance of the class
        """
        super().__init__()
        self.classifier = classifier
        self.models     = models
        if classifier is None or models is None:
            self.text_head = None
        else:
            stack_dim = self.classifier.head.out_features + self.models[min(self.models)].head.out_features
            self.text_head = SimpleHead(stack_dim, self.models[min(self.models)].head.out_features)
            self.fogr_head = SimpleHead(stack_dim, self.classifier.head.out_features)
        self.params = {'type': 'DHCOCR'}

    def convert_widths(self, w, max_width):
        """
        Converts an input widths (in pixel columns) to output widths (in
        the output tensor). Returned as tensor.

        :param w: tensor or array containing a list of width
        :return: long tensor containing the converted widths
        """
        return self.models[self.params['models'][0]].convert_widths(w, max_width)

    def save(self, folder):
        """
        Saves the classifier and OCR models to a folder.
        Creates this folder if needed.

        :param folder: destination folder
        """
        self.classifier.save(join(folder, 'classifier'))
        self.text_head.save(join(folder,  'text-head'))
        self.fogr_head.save(join(folder,  'fogr-head'))
        for n in self.models:
            self.models[n].save(join(folder, '%s' % n))
        self.params['models'] = [x for x in self.models]
        with open(join(folder, 'params.json'), 'wt') as f:
            json.dump(self.params, f, indent=4)

    def to(self, device):
        """
        Overriding Pytorch's to() method, as it doesn't cope well
        with maps.

        :param device: device to send the model to
        :return: the instance on the new device
        """
        self.classifier = self.classifier.to(device)
        self.fogr_head  = self.fogr_head.to(device)
        self.text_head  = self.text_head.to(device)
        for m in self.models:
            self.models[m] = self.models[m].to(device)
        return self

    def parameters(self):
        """
        Overriding Pytorch's parameters() method, as it doesn't cope well
        with maps.

        :return: all trainable parameters of the COCR
        """
        res  = [x for x in self.classifier.parameters()]
        res += [x for x in self.fogr_head.parameters()]
        res += [x for x in self.text_head.parameters()]
        for n in self.models:
            res += [x for x in self.models[n].parameters()]
        return res

    def load_from_folder(folder):
        """
        Static method. Loads an instance of the class from a specific
        folder.

        :param folder: source folder
        :return: a COCR
        """
        net = DHCOCR(None, None)
        net.load(folder)
        return net

    def load(self, folder):
        """
        Loads the classifier and OCR models from a folder.

        :param folder: source folder
        """
        with open(join(folder, 'params.json'), 'rt') as f:
            self.params = json.load(f)
        self.classifier = load_network(join(folder, 'classifier'))
        self.models = {n: load_network(join(folder, '%d' % n)) for n in self.params['models']}
        stack_dim = self.classifier.head.out_features + self.models[min(self.models)].head.out_features
        self.text_head = load_network(join(folder, 'text-head'))
        self.fogr_head = load_network(join(folder, 'fogr-head'))


    def get_optimizers(self, folder=None):
        """
        Returns an array containing optimizers for each part of the
        COCR - if folder is not none, then the optimizers' state dict
        stored in the folder are loaded.

        :param folder: source folder
        :return: an array containing several optimizers
        """
        res = [
            torch.optim.Adam(self.classifier.parameters(), lr=0.001),
            torch.optim.Adam(self.text_head.parameters(), lr=0.001),
            torch.optim.Adam(self.fogr_head.parameters(), lr=0.001)
        ] + [
            torch.optim.Adam(self.models[n].parameters(), lr=0.001) for n in sorted(self.models)
        ]
        if folder is not None:
            res[0].load_state_dict(torch.load(join(folder, 'classifier', 'optimizer.pth')))
            try:
                res[1].load_state_dict(torch.load(join(folder, 'text-head', 'optimizer.pth')))
                res[2].load_state_dict(torch.load(join(folder, 'fogr-head', 'optimizer.pth')))
            except: pass
            for i, n in enumerate(sorted(self.models)):
                res[i+3].load_state_dict(torch.load(join(folder, '%d' % n, 'optimizer.pth')))
        return res

    def save_optimizers(self, optimizers, folder):
        """
        Save the optimizer in a given folder.

        :param folder: destination folder
        """
        torch.save(optimizers[0].state_dict(), join(folder, 'classifier', 'optimizer.pth'))
        torch.save(optimizers[1].state_dict(), join(folder, 'text-head', 'optimizer.pth'))
        torch.save(optimizers[2].state_dict(), join(folder, 'fogr-head', 'optimizer.pth'))
        for i, n in enumerate(sorted(self.models)):
            torch.save(optimizers[i+1].state_dict(), join(folder, '%d' % n, 'optimizer.pth'))

    def forward(self, x):
        """
        Processes an input batch

        :param x: input batch
        :return: the network's output, ready to be convered to a string
        """
        scores = F.softmax(self.classifier(x), dim=2)
        txt = 0
        for n in self.models:
            s = scores[:, :, n].unsqueeze(-1)
            y = self.models[n](x)
            txt += y * s
        cat = torch.cat([scores, txt], dim=2)
        return self.text_head(cat), self.fogr_head(cat)


class PCOCR(torch.nn.Module):
    """
    PCOCR stands for Partial Combined OCR. This system combines multiple OCR
    parts, weighting their outputs locally based on font group classification
    scores.
    """
    def __init__(self, classifier_path, baseline_path, font_groups, common_backbone, common_embed, common_rnn, common_head):
        super().__init__()
        self.classifier = load_network(classifier_path)

        unique_baseline = load_network(baseline_path)
        if common_backbone:
            self.backbone = unique_baseline.backbone
        else:
            self.backbone = {n: load_network(baseline_path).backbone for n in font_groups} # not optimized, but... well...

        if common_embed:
            self.embed = unique_baseline.embed
        else:
            self.embed = {n: load_network(baseline_path).embed for n in font_groups}

        if common_rnn:
            self.rnn = unique_baseline.rnn
        else:
            self.rnn = {n: load_network(baseline_path).rnn for n in font_groups}

        if common_head:
            self.head = unique_baseline.head
        else:
            self.head = {n: load_network(baseline_path).head for n in font_groups}

        # ~ self.act =

        self.params     = {
            'type': 'PCOCR',
            'common-backbone': common_backbone,
            'common-embed': common_embed,
            'common-rnn': common_rnn,
            'common-head': common_head
        }

    def forward(self, x):
        scores = F.softmax(self.classifier(x), dim=2)

        if self.params['common-backbone']:
            x = self.backbone(x)
        else:
            y = 0
            for n in self.models:
                y += self.backbone[n](x) * scores[:, :, n].unsqueeze(-1)
            x = y
        x = self.act(x)

        if self.params['common-embed']:
            x = self.embed(x)
        else:
            y = 0
            for n in self.models:
                y += self.embed[n](x) * scores[:, :, n].unsqueeze(-1)
            x = y


        x = self.backbone(x)

        x = torch.mean(x, axis=2)
        x = x.permute(2, 0, 1)
        x = self.embed(x)
        x, _ = self.rnn(x)
        x = self.head(x)


        res = 0
        for n in self.models:
            s = scores[:, :, n].unsqueeze(-1)
            # ~ if torch.max(s)<0.1:
                # ~ continue
            y = self.models[n](x)
            res += y * s
        return res
