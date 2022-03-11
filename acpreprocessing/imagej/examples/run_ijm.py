from acpreprocessing.imagej import ImageJRunner

imagej_path = '/home/samk/Fiji.app/ImageJ-linux64'

hello_input = {
    'script_type': 'macro',
    'script_path': '/home/samk/axonal_connectomics/imagej/ijm/hello.ijm',
    'args': {
        'string': 'Hello World',
        'num_prints': 5,
    }
}

example_input = {
    'script_type': 'macro',
    'script_path': '/home/samk/axonal_connectomics/imagej/ijm/example.ijm',
    'args': {
        'tiff_path': '/ispim1_data/PoojaB/487748_48_NeuN_NFH_488_25X_0.5XPBS/global_l40_Pos001',
        'gif_path': '/ispim1_data/PoojaB/487748_48_NeuN_NFH_488_25X_0.5XPBS/processed/ds_gifs/global_l40_Pos001'
    }
}

if __name__ == '__main__':
    ijm_runner = ImageJRunner(hello_input, imagej_path=imagej_path)
    ijm_runner.print_script()
    ijm_runner.run()