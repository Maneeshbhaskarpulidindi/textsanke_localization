from tensorboardX import SummaryWriter
from util.misc import mkdirs

class LogSummary(object):

    def __init__(self, log_path):

        mkdirs(log_path)
        self.writer = SummaryWriter(log_path)

    def write_scalars(self, scalar_dict, n_iter, tag=None):

        for name, scalar in scalar_dict.items():
            if tag is not None:
                name = '/'.join([tag, name])
            self.writer.add_scalar(name, scalar, n_iter)

    def write_hist_parameters(self, net, n_iter):
        for name, param in net.named_parameters():
            self.writer.add_histogram(name, param.clone().cpu().numpy(), n_iter)

    def write_image(self, name, image, n_iter):
        # Image is expected to be a numpy array in HWC format.
        # Tensorboard expects CHW for add_image. 
        # Convert HWC to CHW:
        if len(image.shape) == 3 and image.shape[-1] in [1, 3, 4]:
            image = image.transpose((2, 0, 1))
        self.writer.add_image(name, image, n_iter)





