import torch
from torch.utils.data import Dataset


class OCRCorrectionDataset(Dataset):
    '''Dataset containing OCR and GT for OCR post-correction.'''

    def __init__(self, ocr_encodings, gt_encodings):
        self.ocr_encodings = ocr_encodings
        self.gt_encodings = gt_encodings
        assert ocr_encodings.shape == gt_encodings.shape, 'OCR and GT need to have the same dimensions.'

    def __len__(self):
        return len(self.ocr_encodings)

    def __getitem__(self, idx):
        return torch.tensor([self.ocr_encodings[idx], self.gt_encodings[idx]])
