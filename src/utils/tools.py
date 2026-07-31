import torch
import numpy as np


def build_label_map(normal_prompt: str, accident_prompt: str):
    normal_prompt = str(normal_prompt).strip()
    accident_prompt = str(accident_prompt).strip()
    if not normal_prompt or not accident_prompt:
        raise ValueError("Normal and Accident prompts must be non-empty")
    if normal_prompt.casefold() == accident_prompt.casefold():
        raise ValueError("Normal and Accident prompts must be different")
    return {
        "Normal": normal_prompt,
        "Accident": accident_prompt,
    }


def get_batch_label(texts, prompt_text, label_map: dict):
    label_vectors = torch.zeros(0)

    # Hàm ẩn giúp làm sạch text: Chống lỗi tuple/list, chữ hoa/thường, khoảng trắng
    def clean_text(t):
        if isinstance(t, (list, tuple)):
            t = t[0]
        return str(t).strip().lower()

    if len(label_map) != 7:
        if len(label_map) == 2:
            for text in texts:
                clean_t = clean_text(text)
                label_vector = torch.zeros(2)
                
                # FIX: So sánh chuẩn hóa không phân biệt hoa/thường
                if clean_t == 'normal':
                    label_vector[0] = 1
                else:
                    label_vector[1] = 1
                    
                label_vector = label_vector.unsqueeze(0)
                label_vectors = torch.cat([label_vectors, label_vector], dim=0)
        else:
            for text in texts:
                clean_t = clean_text(text)
                label_vector = torch.zeros(len(prompt_text))
                
                # FIX: Tìm key không phân biệt hoa/thường
                matched_key = next((k for k in label_map.keys() if clean_text(k) == clean_t), None)
                if matched_key:
                    label_text = label_map[matched_key]
                    if label_text in prompt_text:
                        label_vector[prompt_text.index(label_text)] = 1

                label_vector = label_vector.unsqueeze(0)
                label_vectors = torch.cat([label_vectors, label_vector], dim=0)
    else:
        for text in texts:
            clean_t = clean_text(text)
            label_vector = torch.zeros(len(prompt_text))
            labels = clean_t.split('-')
            
            for label in labels:
                clean_l = label.strip()
                matched_key = next((k for k in label_map.keys() if clean_text(k) == clean_l), None)
                if matched_key:
                    label_text = label_map[matched_key]
                    if label_text in prompt_text:
                        label_vector[prompt_text.index(label_text)] = 1
            
            label_vector = label_vector.unsqueeze(0)
            label_vectors = torch.cat([label_vectors, label_vector], dim=0)

    return label_vectors

def get_prompt_text(label_map: dict):
    prompt_text = []
    for v in label_map.values():
        prompt_text.append(v)

    return prompt_text

def get_batch_mask(lengths, maxlen):
    batch_size = lengths.shape[0]
    mask = torch.empty(batch_size, maxlen)
    mask.fill_(0)
    for i in range(batch_size):
        if lengths[i] < maxlen:
            mask[i, lengths[i]:maxlen] = 1
    
    return mask.bool()

def random_extract(feat, t_max):
   r = np.random.randint(feat.shape[0] - t_max)
   return feat[r : r+t_max, :]

def uniform_extract(feat, t_max, avg: bool = True):
    new_feat = np.zeros((t_max, feat.shape[1])).astype(np.float32)
    r = np.linspace(0, len(feat), t_max+1, dtype=np.int32)
    if avg == True:
        for i in range(t_max):
            if r[i]!=r[i+1]:
                new_feat[i,:] = np.mean(feat[r[i]:r[i+1],:], 0)
            else:
                new_feat[i,:] = feat[r[i],:]
    else:
        r = np.linspace(0, feat.shape[0]-1, t_max, dtype=np.uint16)
        new_feat = feat[r, :]
            
    return new_feat

def pad(feat, min_len):
    clip_length = feat.shape[0]
    if clip_length <= min_len:
       return np.pad(feat, ((0, min_len - clip_length), (0, 0)), mode='constant', constant_values=0)
    else:
       return feat

def process_feat(feat, length, is_random=False):
    clip_length = feat.shape[0]
    if feat.shape[0] > length:
        if is_random:
            return random_extract(feat, length), length
        else:
            return uniform_extract(feat, length), length
    else:
        return pad(feat, length), clip_length

def process_split(feat, length):
    clip_length = feat.shape[0]
    if clip_length < length:
        return pad(feat, length), clip_length
    else:
        split_num = int(clip_length / length) + 1
        for i in range(split_num):
            if i == 0:
                split_feat = feat[i*length:i*length+length, :].reshape(1, length, feat.shape[1])
            elif i < split_num - 1:
                split_feat = np.concatenate([split_feat, feat[i*length:i*length+length, :].reshape(1, length, feat.shape[1])], axis=0)
            else:
                split_feat = np.concatenate([split_feat, pad(feat[i*length:i*length+length, :], length).reshape(1, length, feat.shape[1])], axis=0)

        return split_feat, clip_length
