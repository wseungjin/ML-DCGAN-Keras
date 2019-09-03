from networks import DCGAN
import argparse
from utills import *

"""parsing and configuration"""

def parse_args():
    desc = "Tensorflow implementation DCGAN"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='train', help='train or test ?')
    parser.add_argument('--dataset', type=str, default='human_faces', help='[anime_faces / simpsons_faces_nm / simpsons_faces_sp /custom_dataset]')

    parser.add_argument('--epoch', type=int, default=30, help='The number of epochs to run')
    parser.add_argument('--iteration', type=int, default=10000, help='The number of training iterations')
    parser.add_argument('--batch_size', type=int, default=100, help='The size of batch per gpu')
    parser.add_argument('--ch', type=int, default=256, help='base channel number per layer')

    parser.add_argument('--print_freq', type=int, default=1000, help='The number of image_print_freqy')
    parser.add_argument('--save_freq', type=int, default=10000, help='The number of ckpt_save_freq')

    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for generator')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for discriminator')

    parser.add_argument('--z_dim', type=int, default=128, help='Dimension of noise vector')

    parser.add_argument('--gan_type', type=str, default='gan', help='dcgan')
    
    parser.add_argument('--img_size', type=int, default=64, help='The size of image')
    parser.add_argument('--sample_num', type=int, default=64, help='The number of sample images')

    parser.add_argument('--test_num', type=int, default=10, help='The number of images generated by the test')

    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')
    parser.add_argument('--sample_dir', type=str, default='samples',
                        help='Directory name to save the samples on training')
    parser.add_argument('--model_dir', type=str, default='models',
                        help='Directory name to save the model')

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)

    # --result_dir
    check_folder(args.result_dir)

    # --result_dir
    check_folder(args.log_dir)

    # --sample_dir
    check_folder(args.sample_dir)

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args


"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    gan = DCGAN(args)

    # build graph
    gan.build_model()

    if gan.phase == 'train' :
        # launch the graph in a session
        gan.train()
        # visualize learned generator

        print(" [*] Training finished!")

    if gan.phase == 'test' :
        gan.test()
        print(" [*] Test finished!")
    
    
main()
