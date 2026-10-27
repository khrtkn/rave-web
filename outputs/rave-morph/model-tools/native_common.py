"""Pure native inference helpers; no source files, catalogs or build side effects."""
import numpy as np
import onnxruntime as ort


def session(data):
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(data, sess_options=options, providers=['CPUExecutionProvider'])


def states_for(sess):
    return {value.name: np.zeros(value.shape, dtype=np.float32)
            for value in sess.get_inputs() if value.name.startswith('state_')}


def errors(a, b):
    difference = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return {'max_abs': float(np.max(np.abs(difference))),
            'rms': float(np.sqrt(np.mean(difference * difference)))}
