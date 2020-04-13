import models.backbones.vgg
import models.backbones.squeezeNet

def get_block(config):
    if('blockName' in config):
        if(config['blockName'] == 'squeezeNet'):
            return squeezeNet.fire_layer
        else:
            return vgg.vgg_block
    else:
        return vgg.vgg_block

def get_backbone(config):
    if('blockName' in config):
        if(config['blockName'] == 'squeezeNet'):
            return squeezeNet.squeezeNet_backbone
        else:
            return vgg.vgg_backbone
    else:
        return vgg.vgg_backbone