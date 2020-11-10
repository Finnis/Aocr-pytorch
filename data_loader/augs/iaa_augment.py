import imgaug
import imgaug.augmenters as iaa


class AugmenterBuilder(object):
    def __init__(self):
        self.sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    def build(self, args, root=True):
        if args is None or len(args) == 0:
            return None
        elif isinstance(args, list):
            if root:
                sequence = [self.sometimes(self.build(value, root=False)) for value in args]
                return iaa.Sequential(sequence)
            else:
                return getattr(iaa, args[0])(*[self._to_tuple_if_list(a) for a in args[1:]])
        elif isinstance(args, dict):
            cls = getattr(iaa, args['type'])
            return cls(**{k: self._to_tuple_if_list(v) for k, v in args['args'].items()})
        else:
            raise RuntimeError('unknown augmenter arg: ' + str(args))
    
    @staticmethod
    def _to_tuple_if_list(obj):
        if isinstance(obj, list):
            return tuple(obj)
        return obj


class IaaAugment(object):
    def __init__(self, augmenter_args):
        self.aug = AugmenterBuilder().build(augmenter_args)

    def __call__(self, image):
        
        image = self.aug.augment_image(image)
            
        return image
