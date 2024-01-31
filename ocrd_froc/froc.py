from io import BufferedReader, BufferedWriter
import pickle

import torch    # type: ignore
from torchvision import transforms    # type: ignore
from PIL import Image

import torch.nn.functional as F    # type: ignore

from ocrd_froc.converter import Converter
from ocrd_froc.classmap import ClassMap

class Froc:
    """ Class wrapping type group information and a classifier.

        Attributes
        ----------

        classMap: ClassMap
            Maps class names to indices corresponding to what the network
            outputs.
        network: PyTorch network
            Classifier
        dev: str
            Device on which the data must be processed

    """

    def __init__(self, groups, charmap, selocr, cocr, device=None):
        """ Constructor of the class.

            Parameters
            ----------

            groups: map string to int
                Maps names to IDs with regard to the network outputs;
                note that several names can point to the same ID, but
                the inverse is not possible.
            network: PyTorch network
                Classifier
            device: str
                Device on which the data has to be processed; if not set,
                then either the cpu or cuda:0 will be used.

        """
        self.converter = Converter(charmap)
        self.classMap = ClassMap(groups)

        self.selocr = selocr
        self.classifier = selocr.classifier

        self.cocr = cocr

        if device is None:
            self.dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.dev = device

        self.selocr.to(self.dev)
        self.cocr.to(self.dev)


    @classmethod
    def load(cls, inp):
        """ Loads a type groups classifier from a file

            Parameters
            ----------
            input: string or file
                File or path to the file from which the instance has to
                be loaded.

        """

        if isinstance(inp, str):
            with open(inp, 'rb') as f:
                return cls.load(f)
        if not isinstance(inp, BufferedReader):
            raise ValueError(f'Froc.load() requires a string or a file, this is neither: {inp}')
        res = pickle.load(inp)
        # If trained with CUDA and loaded on a device without CUDA
        res.dev = torch.device(res.dev if torch.cuda.is_available() else "cpu")
        res.selocr.to(res.dev)
        res.cocr.to(res.dev)
        return res

    def save(self, output):
        """ Stores the instance to a file

            Parameters
            ----------
                output: string or file
                    File or path to the file to which the instane has to
                    be stored.
        """

        if isinstance(output, str):
            with open(output, 'wb') as f:
                self.save(f)
                return
        if not isinstance(output, BufferedWriter):
            raise ValueError(f'Froc.save() requires a string or a file, this is neither: {output}')
        # Moving the network to the cpu so that it can be reloaded on
        # machines which do not have CUDA available.
        self.selocr.to("cpu")
        self.cocr.to("cpu")
        pickle.dump(self, output)
        self.selocr.to(self.dev)
        self.cocr.to(self.dev)

    def run(self, pil_image, method='adaptive', fast_cocr=True, adaptive_treshold=95, classification_result=None):

        if method in ('SelOCR', 'adaptive') and not classification_result:
            raise ValueError(f"Froc.run(): if method is SelOCR or adaptive, classification_result is required")

        tns = self.preprocess_image(pil_image)

        if method == 'SelOCR':
            out = self.run_selocr(tns, classification_result)
        elif method == 'COCR':
            out = self.run_cocr(tns, fast_cocr)
        else:
            out = self.run_adaptive(tns, classification_result, fast_cocr, adaptive_treshold)

        # constrain to image width, expand to batch format (batch size 1)
        base_width = [tns.shape[2]]

        out = torch.softmax(out, 2)
        scores, res = torch.max(out[:, :, :], 2)

        res, score = self.converter.decode(res, scores, base_width=base_width)

        # squeeze batch dimension
        res = res[0]
        score = score[0]
        return res, score


    def classify(self, pil_image):
        trans = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])

        if pil_image.size[1]!=32:
            ratio = 32 / pil_image.size[1]
            width = int(pil_image.size[0] * ratio)
            pil_image = pil_image.resize((width, 32), Image.Resampling.LANCZOS)

        tns = trans(pil_image).to(self.dev).unsqueeze(0)
        out = self.classifier(tns)
        score = out.mean(axis=1)[0]
        score = F.softmax(score, dim=0)

        res = {}
        for cl in self.classMap.cl2id:
            cid = self.classMap.cl2id[cl]
            if cid == -1:
                continue
            res[cl] = score[cid].item()
        return res


    def preprocess_image(self, pil_image):
        if pil_image.size[1]!=32:
            ratio = 32 / pil_image.size[1]
            width = int(pil_image.size[0] * ratio)
            try:
                pil_image = pil_image.resize((width,32), Image.Resampling.LANCZOS)
            except Exception as e:
                print('Cannot resize')
                raise e

        trans = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])

        tns = trans(pil_image)
        tns = tns.to(self.dev)

        return tns

    def run_selocr(self, tns, classification_result):
        max_cl = max(classification_result, key=classification_result.get)
        max_idx = self.classMap.cl2id[max_cl]

        tns = torch.unsqueeze(tns, 0)
        with torch.no_grad():
            self.selocr.eval()

            out = self.selocr(tns, max_idx)
            out = out.transpose(0,1)

            return out

    def run_cocr(self, tns, fast_cocr):
        tns = torch.unsqueeze(tns, 0)

        with torch.no_grad():
            self.cocr.eval()

            out = self.cocr(tns, fast_cocr)
            out = out.transpose(0,1)

            return out


    def run_adaptive(self, tns, classification_result, fast_cocr, adaptive_treshold):
        if max(classification_result.values()) > adaptive_treshold / 100:
            return self.run_selocr(tns, classification_result)
        else:
            return self.run_cocr(tns, fast_cocr)
