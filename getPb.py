from model import getModel
import tensorflow as tf

useSimpleSave = True


def getPb():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        modelX, modelY, modelLoss, prediction = getModel()
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint("Models/"))
        if useSimpleSave:
            tf.saved_model.simple_save(
                sess,
                "Models/Pb/1",
                inputs={
                    "x": modelX
                },
                outputs={
                    "y": prediction
                }
            )
        else:
            tensorInfoX = tf.saved_model.utils.build_tensor_info(modelX)
            tensorInfoPreds = tf.saved_model.utils.build_tensor_info(prediction)
            exportPath = "Models/Pb/1"
            builder = tf.saved_model.builder.SavedModelBuilder(exportPath)
            predictSign = (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={
                        "x": tensorInfoX
                    },
                    outputs={
                        "y": tensorInfoPreds
                    },
                    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
                )
            )
            legacyInitOp = tf.group(
                tf.tables_initializer(),
                name="legacyInitOp"
            )
            builder.add_meta_graph_and_variables(
                sess,
                [tf.saved_model.tag_constants.SERVING],
                signature_def_map={
                    "predict": predictSign,
                },
                legacy_init_op=legacyInitOp
            )
            builder.save()


getPb()

