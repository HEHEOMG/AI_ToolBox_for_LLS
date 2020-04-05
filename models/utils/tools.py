import torch



def save_checkpoint(state, file_name = 'checkpoint.pth'):
    """
    save model checkpoint
    :param state:
    :param is_best:
    :param file_name:
    :return:

    eg:
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, is_best)
    """
    torch.save(state, file_name)


def load_checkpoint(model, checkpoint_path=None, optimizer= None):
    """
    load model checkpoint
    :param model:
    :param checkpoint_path: 
    :param optimizer:
    :return:
    """
    if checkpoint_path != None:
        model_ckpt = torch.load(checkpoint_path)
        model.load_state_dict(model_ckpt['state_dict'])
        print('loading checkpoint！')
    if optimizer !=None:
        optimizer.load_state_dict(model_ckpt['optimizer'])