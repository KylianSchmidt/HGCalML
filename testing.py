import tensorflow as tf
import uproot
import numpy as np
import awkward as ak


def get_normal_hits() -> (np.ndarray, np.ndarray):
    prefix = "Events/Output/fHits/fHits."
    d, hits = {}, []

    with uproot.open("./nntr_data/normal_detector/Raw/Testing.root") as file:
        for keys in ["layerType", "x", "y", "z", "E"]:
            d[keys] = file[prefix+keys].array(library="ak")[0:5]
            d[keys] = d[keys].mask[d["layerType"] > 0]
            hits.append(d[keys][..., np.newaxis])

        hits = ak.concatenate(hits, axis=-1)

    offsets = np.cumsum(
        np.append(
            0, np.array(
                ak.to_list(
                    ak.num(hits, axis=1)))))
    
    keep_index = np.zeros(len(offsets)-1, dtype='bool')

    for i in range(1, len(offsets)):
        if offsets[i-1] != offsets[i]:
            keep_index[i-1] = True
        else:
            print("Removed empty event with ID", i-1)

    hits_unique = hits[keep_index]
    hits_unique = ak.to_numpy(ak.flatten(hits_unique, axis=1))
    offsets = np.unique(offsets)

    return hits_unique, offsets


def garnet(inputs: [tf.Tensor, np.ndarray]):
    x, rs = inputs
    n_propagator = 5
    n_aggregators = 3

    print("Inputs[0]\n", x.shape)
    # F_in: rs(B*H, F) = (B, H, F)
    F_in = tf.RaggedTensor.from_row_splits(x, rs)
    print("F_in\n", F_in.shape)
    # d distance between H vertices and S aggregators
    # d = Dense(B*H, F) = (B*H, S)
    # Dense layer, will not accept ragged tensor
    distance = x[:, 0:1] * tf.constant(np.full((x.shape[0], n_aggregators), 3.0))
    print("Distance\n", distance.shape)
    # Matrix V(d_jk) (B, H, S)
    # TODO replace with tf.exp(-distance**2)
    edge_weights = tf.RaggedTensor.from_row_splits(distance*7, rs)
    print("Edge_weights (rs)\n", edge_weights.shape)
    print("Edge_weights (rs) [0]\n", edge_weights[0:1])
    # f_ij Dense(B*H, F) = (B*H, F)
    # Has same shape as F_in but learned representation therefore new values
    # Dense layer, will not accept ragged tensor
    features_LR = x[:, 0:1] * tf.constant(np.full((x.shape[0], n_propagator), 5.0))
    print("Features_LR (before rowsplits)\n", features_LR.shape)
    features_LR = tf.RaggedTensor.from_row_splits(features_LR, rs)
    print("Features_LR (after rowsplits)\n", features_LR.shape)
    print("Features_LR (after rowsplits) [0]\n", features_LR[0:1])
    # f_tilde: f_ij x V(d_jk) = (B, H, 1, P) x (B, H, S, 1) = (B, H, S, P)
    f_tilde = tf.expand_dims(features_LR, axis=2) * tf.expand_dims(edge_weights, axis=3)
    print("f_tilde\n", f_tilde.shape)
    print("f_tilde[0]\n", f_tilde[0:1])
    # f_tilde_mean and max (B, S, P)
    f_tilde_mean = tf.reduce_mean(f_tilde+1E-10, axis=1)
    f_tilde_max = tf.reduce_max(f_tilde, axis=1)
    print("f_tilde_mean\n", f_tilde_mean.shape)shape 
    # Aggregation of f_tilde (B, S, 2*P))
    f_tilde_mean_max = tf.concat([f_tilde_mean, f_tilde_max], axis=-1)
    print("f_tilde_mean_max\n", f_tilde_mean_max.shape)
    print("f_tilde_mean_max expanded", tf.expand_dims(f_tilde_mean_max, axis=1).shape)
    print("edge weights expanded", tf.expand_dims(edge_weights, axis=3).shape)
    # Return f_updated to the hits: (B, 1, S, 2*P) x (B, H, S, 1)  = (B, H, S, 2*P)
    f_updated = tf.expand_dims(f_tilde_mean_max, axis=1) * tf.expand_dims(edge_weights, axis=3)
    print("f_updated\n", f_updated.shape)
    # Reshape to (B, H, 2*P*S)
    f_updated = f_updated.merge_dims(2, 3)
    print("f_updated reshaped\n", f_updated.shape)
    sampled_event = f_updated[1:2, 0:1, :]
    print("Output f_updated", sampled_event.shape, "\n", sampled_event)
    # TODO Final dense layer


hits = ak.Array([
    [[2.0, 2.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0]],
    [[2.0, 2.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0]],
    [[2.0, 2.0, 2.0, 2.0]],
    [[2.0, 2.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0]]
])
print("Hits (ak)\n(3, None, 4)")
offsets = np.cumsum(
    np.append(
        0, np.array(
            ak.to_list(
                ak.num(hits, axis=1)))))
hits = ak.to_numpy(ak.flatten(hits, axis=1))
print("Offsets\n", offsets.shape)

inputs = [tf.constant(hits), offsets]
garnet(inputs)
