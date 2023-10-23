import tensorflow as tf
import uproot
import numpy as np
import awkward as ak


def get_normal_hits():
    prefix = "Events/Output/fHits/fHits."
    d, hits = {}, []

    with uproot.open("./nntr_data/normal_detector/Raw/Testing.root") as file:
        for keys in ["layerType", "x", "y", "z", "E"]:
            d[keys] = file[prefix+keys].array(library="ak")
            d[keys] = d[keys].mask[d["layerType"] > 0]
            hits.append(d[keys][..., np.newaxis])

        hits = ak.concatenate(hits, axis=-1)

    offsets = np.cumsum(
        np.append(
            0, np.array(
                ak.to_list(
                    ak.num(hits, axis=1)))))
    return hits, offsets


def garnet(inputs: []):
    x, rs = inputs
    n_propagator = 5
    print("Inputs[0]\n", x.shape)
    # F_in: rs(B*H, F) = (B, H, F)
    F_in = tf.RaggedTensor.from_row_splits(x, rs)
    print("F_in\n", F_in.shape)
    # d distance between H vertices and S aggregators
    # d = Dense(B*H, F) = (B*H, S)
    # Dense layer, will not accept ragged tensor
    dense_kernel = tf.constant(np.array([[2.0, 2.0, 2.0]]))
    print("dense_kernel\n", dense_kernel.shape)
    distance = x[:, 0:1] * dense_kernel
    print("Distance\n", distance.shape)
    # Matrix V(d_jk) (B, H, S)
    edge_weights = tf.RaggedTensor.from_row_splits(tf.exp(-distance**2), rs)
    print("Edge_weights (rs)\n", edge_weights.shape)
    # f_ij Dense(B*H, F) = (B*H, F)
    # Has same shape as F_in but learned representation therefore new values
    # Dense layer, will not accept ragged tensor
    features_LR = x[:, 0:1] * tf.constant(
        np.full((x.shape[0], n_propagator), 5.0))
    print("Features_LR (before rowsplits)\n", features_LR.shape)
    features_LR = tf.RaggedTensor.from_row_splits(features_LR, rs)
    print("Features_LR (after rowsplits)\n", features_LR.shape)
    # f_tilde: f_ij x V(d_jk) = (B, H, 1, P) x (B, H, S, 1) = (B, H, S, P)
    f_tilde = tf.expand_dims(features_LR, axis=2) * tf.expand_dims(edge_weights, axis=3)
    print("f_tilde\n", f_tilde.shape)
    # f_tilde_average and max (B, S, P)
    f_tilde_average = tf.reduce_mean(f_tilde, axis=1)
    f_tilde_max = tf.reduce_max(f_tilde, axis=1)
    print("f_tilde_average\n", f_tilde_average.shape)
    # Aggregation of f_tilde (B, S, 2*P))
    f_tilde_mean_max = tf.concat([f_tilde_average, f_tilde_max], axis=-1)
    print("f_tilde_mean_max\n", f_tilde_mean_max.shape)
    print("filde_mean_max expanded", tf.expand_dims(f_tilde_mean_max, axis=1).shape)
    print("edge weights expanded", tf.expand_dims(edge_weights, axis=3).shape)
    # Return f_updated to the hits: (B, 1, S, 2*P) x (B, H, S, 1)  = (B, H, S, 2*P)
    f_updated = tf.expand_dims(f_tilde_mean_max, axis=1) * tf.expand_dims(edge_weights, axis=3)
    print("f_updated\n", f_updated.shape)
    print(f_updated)
    # Reshape to (B, H, 2*P*S)
    f_updated = f_updated.merge_dims(2, 3)
    print("f_updated reshaped\n", f_updated.shape)


hits = ak.Array([
    [[2.0, 2.0, 2.0, 2.0], [3.0, 3.0, 3.0, 3.0]],
    [[5.0, 5.0, 5.0, 5.0], [7.0, 7.0, 7.0, 7.0], [11.0, 11.0, 11.0, 11.0]],
    [[13.0, 13.0, 13.0, 13.0]]
])
print("Hits (ak)\n(3, None, 4)")
offsets = np.cumsum(
    np.append(
        0, np.array(
            ak.to_list(
                ak.num(hits, axis=1)))))
print("Offsets\n", offsets.shape)

hits = ak.to_numpy(ak.flatten(hits, axis=1))
print("Hits (np)\n", hits.shape)

inputs = [
    tf.constant(hits),
    offsets
]
garnet(inputs)
