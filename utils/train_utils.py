import lightning as PL
import torch


def load_finetune_ckpt( model,state_dict,strict_shapes:bool=True) -> None:
    adapt_shapes = strict_shapes
    if not adapt_shapes:
        cur_model_state_dict = model.state_dict()
        unmatched_keys = []
        for key, param in state_dict.items():
            if key in cur_model_state_dict:
                new_param = cur_model_state_dict[key]
                if new_param.shape != param.shape:
                    unmatched_keys.append(key)
                    print('| Unmatched keys: ', key, new_param.shape, param.shape)
        for key in unmatched_keys:
            del state_dict[key]
    model.load_state_dict(state_dict, strict=False)




def load_pre_train_model(finetune_ckpt_path:str,finetune_load_params:list):
    pre_train_ckpt_path = finetune_ckpt_path
    blacklist = finetune_load_params
    # whitelist=hparams.get('pre_train_whitelist')
    if blacklist is None:
        blacklist = []
    # if whitelist is  None:
    #     raise RuntimeError("")

    if pre_train_ckpt_path is not None:
        ckpt = torch.load(pre_train_ckpt_path)
        # if ckpt.get('category') is None:
        #     raise RuntimeError("")

        # if isinstance(self.model, CategorizedModule):
        #     self.model.check_category(ckpt.get('category'))

        state_dict = {}
        for i in ckpt['state_dict']:
            # if 'diffusion' in i:
            # if i in rrrr:
            #     continue
            skip = False
            for b in blacklist:
                if i.startswith(b):
                    skip = True
                    break

            if skip:
                continue

            state_dict[i] = ckpt['state_dict'][i]
            print(i)
        return state_dict
    else:
        raise RuntimeError("")




def get_need_freeze_state_dict_key(frozen_params, model_state_dict) -> list:
    key_list = []
    for i in frozen_params:
        for j in model_state_dict:
            if j.startswith(i):
                key_list.append(j)
    return list(set(key_list))



def freeze_params(model,frozen_params) -> None:
    model_state_dict = model.state_dict().keys()
    freeze_key = get_need_freeze_state_dict_key(frozen_params=frozen_params,model_state_dict=model_state_dict)

    for i in freeze_key:
        params = model.get_parameter(i)

        params.requires_grad = False


def unfreeze_all_params(model) -> None:
    for i in model.parameters():
        i.requires_grad = True


