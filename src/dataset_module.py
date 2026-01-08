import os
from torch.utils.data import Dataset
from torch_geometric.data import DataLoader as DataLoaderGeo
from datasets import load_dataset
from torchvision import transforms

class MoleculeGeneration(Dataset):
    def __init__(self,
                 args,
                 tokenizer,
                 dataset_name_or_path='duongttr/LPM-24-extend', 
                 split='train',
                 input_max_length=512,
                 output_max_length=512):
        super().__init__()
        num_cores = os.cpu_count()
        self.dataset = load_dataset(dataset_name_or_path, split=split, use_auth_token=True, num_proc=num_cores)
        
        # preprocessing data
        if 'LPM-24' in dataset_name_or_path:
            self.dataset = self.dataset.filter(lambda sample: sample['selfies'] != '', num_proc=num_cores)
            
        self.is_lpm_24 = 'LPM-24' in dataset_name_or_path
            
        self.tokenizer = tokenizer
        self.input_max_length = input_max_length
        self.output_max_length = output_max_length

        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, index, add_padding=True):
        sample = self.dataset[index]
        
        if self.is_lpm_24:
            sample_selfies = sample['selfies']
            sample_caption = sample['caption']
            sample_image = sample['image']
        else:
            sample_selfies = sample['SELFIES']
            sample_caption = sample['description']
            sample_image = sample['image']

        # Language-to-molecule (SELFIES) prompt
        task_definition = (
            'Definition: You are given a molecule description in English. '
            'Your job is to generate the corresponding molecule in SELFIES representation.\n\n'
        )
        task_input = f'Now complete the following example -\nInput: <bop>{sample_caption}<eop>\nOutput: '
        
        model_input = task_definition + task_input
        
        if add_padding:
            input = self.tokenizer(
                model_input,
                add_special_tokens=True,
                max_length=self.input_max_length,
                padding = 'max_length',
                truncation = True,
                return_attention_mask = True,
                return_tensors='pt'
            )
            
            # For lang2mol, the target is the SELFIES string
            output = self.tokenizer(
                sample_selfies,
                add_special_tokens=True,
                max_length=self.output_max_length,
                padding = 'max_length',
                truncation = True,
                return_attention_mask = True,
                return_tensors='pt'
            )
        else:
            input = self.tokenizer(
                model_input,
                add_special_tokens=True,
                return_attention_mask = True,
                return_tensors='pt'
            )
            
            # For lang2mol, the target is the SELFIES string
            output = self.tokenizer(
                sample_selfies,
                add_special_tokens=True,
                return_attention_mask = True,
                return_tensors='pt'
            )
        
        input_ids = input['input_ids'].flatten()
        attention_mask = input['attention_mask'].flatten()
        labels = output['input_ids'].flatten()
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'images': self.image_transform(sample_image.convert('L').convert('RGB')),
            'selfies': sample_selfies,
            'caption': sample_caption
        }

        
class MoleculeGeneration_InferLPM24(Dataset):
    def __init__(self,
                 args,
                 tokenizer,
                 dataset_name_or_path='duongttr/LPM-24-eval-caption', 
                 split='train',
                 input_max_length=512,
                 output_max_length=512):
        super().__init__()
        self.dataset = load_dataset(dataset_name_or_path, split=split, use_auth_token=True)
        
        # preprocessing data
        if 'LPM-24' in dataset_name_or_path:
            self.dataset = self.dataset.filter(lambda sample: sample['selfies'] != '') # remove invalid selfies
            
        self.is_lpm_24 = 'LPM-24' in dataset_name_or_path
            
        self.tokenizer = tokenizer
        self.input_max_length = input_max_length
        self.output_max_length = output_max_length
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, index):
        sample = self.dataset[index]
        
        
        sample_selfies = sample['selfies']
        sample_smiles = sample['molecule']
        sample_image = sample['image']
        sample_caption = sample['caption']

        task_definition = 'Definition: You are given a molecule description in English. Your job is to generate the corresponding molecule in SELFIES representation.\n\n'
        task_input = f'Now complete the following example -\nInput: <bop>{sample_caption}<eop>\nOutput: '
        
        model_input = task_definition + task_input
        
        
        input = self.tokenizer(
            model_input,
            add_special_tokens=True,
            max_length=self.input_max_length,
            padding = 'max_length',
            truncation = True,
            return_attention_mask = True,
            return_tensors='pt'
        )
        
        output = self.tokenizer(
            sample_selfies,
            add_special_tokens=True,
            max_length=self.output_max_length,
            padding = 'max_length',
            truncation = True,
            return_attention_mask = True,
            return_tensors='pt'
        )
        
        input_ids = input['input_ids'].flatten()
        attention_mask = input['attention_mask'].flatten()
        labels = output['input_ids'].flatten()
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'selfies': sample_selfies,
            'caption': sample_caption
        }
        

def get_dataloaders(args, tokenizer, batch_size=8, num_workers=4, split='train', task='mol2lang'):
    if task == 'lang2mol':
        dataset = MoleculeGeneration(
            args,
            tokenizer=tokenizer,
            dataset_name_or_path=args.dataset_name_or_path,
            split=split,
            input_max_length=args.eval_max_length,
            output_max_length=args.eval_max_length)

    return DataLoaderGeo(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=(split == 'train'),
        pin_memory=args.pin_memory,
        persistent_workers= args.persistent_workers,
    )