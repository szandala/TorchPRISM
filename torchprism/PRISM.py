
from torch.nn import Conv2d, MaxPool2d
from torch import no_grad
from torch.nn.functional import interpolate

class PRISM:
    _excitations = []
    _hook_handlers = []

    def _excitation_hook(module, input, output):
        PRISM._excitations.append(input[0])

    def register_hooks(model, recursive=False):
        if not recursive and PRISM._hook_handlers:
            print("Warning: hooks can only be registered to one model at once. Please use: `prune_old_hooks()`")
            return

        for i, layer in enumerate(model.children()):
            if list(layer.children()):
                PRISM.register_hooks(layer, recursive=True)
            elif isinstance(layer, MaxPool2d):
                PRISM._hook_handlers.append(layer.register_forward_hook(PRISM._excitation_hook))
            elif isinstance(layer, Conv2d) and layer.stride > (1,1):
                PRISM._hook_handlers.append(layer.register_forward_hook(PRISM._excitation_hook))

    def prune_old_hooks(model):
        for hook in PRISM._hook_handlers:
            hook.remove()
        else:
            print("No hooks to remove")
        PRISM._hook_handlers = []

###############################################

    def _svd(final_excitation, channels = 3):
        print("svd")
        print(final_excitation.shape)
        final_layer_input = final_excitation.permute(0,2,3,1).reshape(-1, final_excitation.shape[1])
        print(final_layer_input.shape)
        normalized_final_layer_input = final_layer_input - final_layer_input.mean(0)
        u, s, v = normalized_final_layer_input.svd(compute_uv=True)
        raw_features = u[:,:channels].matmul(s[:channels].diag())
        print(raw_features.shape)
        x = raw_features.view(final_excitation.shape[0], final_excitation.shape[2], final_excitation.shape[3], 3).permute(0,3,1,2)
        print(x.shape)
        return x

    def _feature_normalization(single_excitation):
        feature_excitation = single_excitation.sum(dim=1).unsqueeze(1)
        # print(f"FE1: {single_excitation.shape}")
        feature_excitation /= feature_excitation.max()
        # print(f"FE2: {feature_excitation.shape}")
        return feature_excitation

    def _upsampling(extracted_features, pre_excitations):
        for e in pre_excitations[::-1]:
            extracted_features = interpolate(extracted_features, size=(e.shape[2], e.shape[3]), mode="bilinear", align_corners=False)
            extracted_features *= PRISM._feature_normalization(e)
        return extracted_features

    def _normalize_to_rgb(features):
        scaled_features = (features - features.mean()) / features.std()
        scaled_features = scaled_features.clip(-1, 1)
        scaled_features = (scaled_features - scaled_features.min()) / (scaled_features.max()-scaled_features.min())
        return scaled_features

    def get_maps():
        if PRISM._excitations:
            desired_shape = PRISM._excitations[0].shape[-2:]
        else:
            print("No data in hooks. Have You used `register_hooks(model)` method?")
            return

        # print(f"PRISM._excitations size: {len(PRISM._excitations)}")
        [print(e.shape) for e in PRISM._excitations]

        with no_grad():
            extracted_features = PRISM._svd(PRISM._excitations.pop(), 3)
            extracted_features = PRISM._upsampling(extracted_features, PRISM._excitations)
            rgb_features_map = PRISM._normalize_to_rgb(extracted_features)

            # prune old PRISM._excitations
            PRISM.reset_excitations()

            return rgb_features_map

    def reset_excitations():
        PRISM._excitations = []

