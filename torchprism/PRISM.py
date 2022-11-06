from torch.nn import Conv2d, MaxPool2d
from torch import no_grad, round, cat
from torch.nn.functional import interpolate
from itertools import chain
from icecream import ic

class PRISM:
    _excitations = []
    _hook_handlers = []
    _is_orig_image = True
    _variances = None

    def get_variances():
        return PRISM._variances

    def _excitation_hook(module, input, output):
        # for better output sharpness we collect input images
        if PRISM._is_orig_image:
            PRISM._excitations.append(input[0])
            PRISM._is_orig_image = False

        PRISM._excitations.append(output)

    def register_hooks(model, recursive=False):
        if not recursive and PRISM._hook_handlers:
            print("Hooks can only be registered to one model at once. Please use: `prune_old_hooks()`")
            return

        for i, layer in enumerate(model.children()):
            if list(layer.children()):
                PRISM.register_hooks(layer, recursive=True)
            elif isinstance(layer, MaxPool2d):
                PRISM._hook_handlers.append(
                    layer.register_forward_hook(PRISM._excitation_hook)
                )
            elif isinstance(layer, Conv2d) and layer.stride > (1, 1):
                PRISM._hook_handlers.append(
                    layer.register_forward_hook(PRISM._excitation_hook)
                )

    def prune_old_hooks(model):
        if not PRISM._hook_handlers:
            print("No hooks to remove")
        for hook in PRISM._hook_handlers:
            hook.remove()

        PRISM._hook_handlers = []
        PRISM._variances = None
    ###############################################

    def _svd(final_excitation):

        final_layer_input = final_excitation.permute(0, 2, 3, 1).reshape(
            -1, final_excitation.shape[1]
        )
        normalized_final_layer_input = final_layer_input - final_layer_input.mean(0)

        u, s, v = normalized_final_layer_input.svd(compute_uv=True)
        PRISM._variances = s**2/sum(s**2)
        raw_features = u[:, :3].matmul(s[:3].diag())

        return raw_features.view(
            final_excitation.shape[0],
            final_excitation.shape[2],
            final_excitation.shape[3],
            3
        ).permute(0, 3, 1, 2)

    def _quantize(maps):
        # h,w,c
        maps = PRISM._normalize_to_rgb(maps).permute(0, 2, 3, 1)
        quant_maps = 0.5 * round(maps / 0.5)
        image_colors = []
        for img in quant_maps:
            colors_set = set()
            for row in img:
                for pixel in row:
                    colors_set.add(pixel.cpu().numpy().tostring())
            image_colors.append(colors_set)

        return quant_maps, image_colors

    def _intersection(maps):
        quant_maps, image_colors = PRISM._quantize(maps)
        common_colors = set.intersection(*image_colors)
        for img in quant_maps:
            for row in img:
                for pixel in row:
                    if pixel.numpy().tostring() not in common_colors:
                        pixel *= 0.0
        return quant_maps.permute(0, 3, 1, 2)

    def _difference(maps):
        quant_maps, image_colors = PRISM._quantize(maps)
        all_colors= set(chain.from_iterable(image_colors))
        exclusive_colors = all_colors - set.intersection(*image_colors)
        for img in quant_maps:
            for row in img:
                for pixel in row:
                    if pixel.cpu().numpy().tostring() not in exclusive_colors:
                        pixel *= 0.0
        return quant_maps.permute(0, 3, 1, 2)


    def _upsampling(extracted_features, pre_excitations):
        for e in pre_excitations[::-1]:
            extracted_features = interpolate(
                extracted_features,
                size=(e.shape[2], e.shape[3]),
                mode="bilinear",
                align_corners=False,
            )
            extracted_features *= e.mean(dim=1, keepdim=True)
        return extracted_features

    def _normalize_to_rgb(features):
        scaled_features = (features - features.mean()) / features.std()
        scaled_features = scaled_features.clip(-1, 1)
        scaled_features = (scaled_features - scaled_features.min()) / (
            scaled_features.max() - scaled_features.min()
        )
        return scaled_features

    def _merge_with_saliency(rgb_features_map, saliency_map):
        saliency_map -= saliency_map.min()
        saliency_map /= saliency_map.max()
        rgb_features_map = cat((rgb_features_map, saliency_map), dim=1)
        return rgb_features_map

    def get_maps(grad_extrap=True, inclusive=False, exclusive=False, saliency_map=None):
        if not PRISM._excitations:
            print("No data in hooks. Have You used `register_hooks(model)` method?")
            return

        with no_grad():
            extracted_features = PRISM._svd(PRISM._excitations.pop())

            if inclusive and exclusive:
                rgb_features_map, _ = PRISM._quantize(extracted_features)
                rgb_features_map = rgb_features_map.permute(0, 3, 1, 2)
            elif exclusive:
                rgb_features_map = PRISM._difference(extracted_features)
            elif inclusive:
                rgb_features_map = PRISM._intersection(extracted_features)
            else:
                rgb_features_map = extracted_features

            if not saliency_map == None:
                rgb_features_map = PRISM._merge_with_saliency(rgb_features_map, saliency_map)
            if grad_extrap:
                rgb_features_map = PRISM._upsampling(
                    rgb_features_map, PRISM._excitations
                )
            rgb_features_map = PRISM._normalize_to_rgb(rgb_features_map)

            # prune old PRISM._excitations
            PRISM.reset_excitations()

            return rgb_features_map

    def reset_excitations():
        PRISM._is_orig_image = True
        PRISM._excitations = []
