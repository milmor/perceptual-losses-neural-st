hparams = {'input_size': (256, 256, 3),
           'batch_size': 4,
           'content_weight': 1e-5,#1e-1
           'style_weight': 5e-9, #5e-6
           'learning_rate': 0.001,
           'test_size': (1280, 1920, 3),
		   'residual_filters': 64,
           'residual_layers': 5,
           'initializer': "glorot_normal",
}
