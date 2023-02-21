import re
import torch

wtf_pattern = re.compile(r'(.[\u02F3\u1D53\u0300\u2013\u032E\u208D\u203F\u0311\u0323\u035E\u031C\u02FC\u030C\u02F9\u0328\u032D\u02F4\u032F\u0330\u035C\u0302\u0327\u0357\u0308\u0351\u0304\u02F2\u0352\u0355\u032C\u030B\u0339\u0301\u02F1\u0303\u0306\u030A\u0325\u0307\u0354\u02F0\u0060\u030d\u0364\u0303]*)', re.UNICODE | re.IGNORECASE)
def split(s):
    return list(wtf_pattern.findall(s))


class Converter(object):
    def __init__(self, letters):
        self.letters = {l:(n+1) for n, l in enumerate(letters)}
        self.n_classes = len(letters)+1
        self.reverse = {self.letters[l]:l for l in self.letters} | {0: ''}
    
    def encode(self, text, pad=True):
        if isinstance(text, str):
            s = split(text)
            return [self.letters[x] for x in s], len(s)
        encodings = []
        lengths   = []
        for item in text:
            e, l = self.encode(item)
            encodings.append(e)
            lengths.append(l)
        if pad:
            mx = np.max(lengths)
            for e in encodings:
                while len(e)<mx:
                    e.append(0)
        return torch.Tensor(encodings).int(), torch.Tensor(lengths).int()
    
    def raw(self, encodings):
        n = len(encodings.shape)
        if n>3:
            raise Exception('Wrong encoding shape, must be 1D, 2D tensor (batch, length), or 3d (batch, length, class scores). You gave:'+str(encodings.shape))
        if n==1:
            l = []
            for i in range(encodings.shape[0]):
                l.append(self.reverse[encodings[i].item()])
            return ''.join(l)
        if n==2:
            res = []
            for i in range(encodings.shape[0]):
                res.append(self.raw(encodings[i]))
            return res
        if n==3:
            return self.raw(torch.argmax(out, 2))
    
    def decode(self, encodings, base_width=None):
        n = len(encodings.shape)
        if n>3:
            raise Exception('Wrong encoding shape, must be 1D, 2D tensor (batch, length), or 3d (batch, length, class scores). You gave:'+str(encodings.shape))
        if n==1:
            l = []
            prev = None
            for i in range(min(encodings.shape[0], base_width)):
                cur = encodings[i]
                if cur!=prev:
                    l.append(self.reverse[cur.item()])
                prev = cur
            return ''.join(l).replace('/PAD/', '')
        if n==2:
            res = []
            for i in range(encodings.shape[0]):
                if base_width is not None:
                    res.append(self.decode(encodings[i], base_width[i]))
                else:
                    res.append(self.decode(encodings[i], encodings[i].shape[0]))
            return res
        if n==3:
            return self.decode(torch.argmax(out, 2), base_width)
