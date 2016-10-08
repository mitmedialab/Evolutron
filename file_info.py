import argparse

import numpy as np
from Bio import SeqIO

import evolutron.networks.thn as nets
from evolutron.trainers.thn import DeepTrainer


def main(filename, **options):

    if '.fasta' in filename:
        count = 0
        lengths = []
        for record in SeqIO.parse(filename, "fasta"):
            count += 1
            lengths.append(len(str(record.seq)))

        lengths = np.asarray(lengths)
        print('File has {0} sequences'.format(count))
        print('Minimum length: {0}'.format(np.min(lengths)))
        print('Maximum length: {0}'.format(np.max(lengths)))
        print('Average length: {0}'.format(np.mean(lengths)))

        if options['graph']:
            import matplotlib.pyplot as plt
            plt.plot(lengths)
            plt.title('Length Distribution')
            plt.xlabel('Seq ID')
            plt.ylabel('Length')

            try:
                plt.show()
            except KeyboardInterrupt:
                return

    elif '.model.' in filename:
        # Load network parameters0
        with np.load(filename) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        filters = param_values[0].shape[0]
        filter_size = param_values[0].shape[2]

        if 'CoHST' in filename:
            net = nets.CoHST(None, filters, filter_size)
        elif 'CoDER' in filename:
            net = nets.CoDER(None, filters, filter_size)
        else:
            if 'b1h' in filename:
                net = nets.ConvZFb1h(None, filters, filter_size)
            elif 'type2p' in filename:
                net = nets.ConvType2p(None, filters, filter_size)
            elif 'm6a' in filename:
                net = nets.ConvM6a(None, filters, filter_size, False)
            else:
                raise NotImplementedError('filename not able to be visualized at this moment.')

        conv_net = DeepTrainer(net)

        conv_net.display_network_info()
    elif '.history.' in filename:
        with np.load(filename) as f:
            try:
                if len(f.files) == 2:
                    train_err_mem = f['train_err_mem'] 
                    val_err_mem = f['val_err_mem']
                    train_acc_mem = ['NA']
                    val_acc_mem = ['NA']
                else :
                    train_err_mem = f['train_err_mem'] 
                    val_err_mem = f['val_err_mem']     
                    train_acc_mem = f['train_acc_mem'] 
                    val_acc_mem = f['val_acc_mem']
            except:
                raise IOError('Invompatible history file')
            
            print('Model was trained for {0} epochs'.format(len(train_err_mem)))
            print('Best training error was: {0}'.format(train_err_mem[-1]))
            print('Best validation error was: {0}'.format(val_err_mem[-1]))
            
            print('Best training accuracy was: {0}'.format(train_acc_mem[-1]))
            print('Best validation accuracy was: {0}'.format(val_acc_mem[-1]))

            if options['graph']:
                import matplotlib.pyplot as plt
                plt.figure(0)
                plt.plot(train_err_mem)
                plt.plot(val_err_mem)
                plt.title('Training Error Curve')
                plt.xlabel('Epochs')
                plt.ylabel('MSE or Cross-entropy')

                plt.figure(1)
                plt.plot(train_acc_mem)
                plt.plot(val_acc_mem)
                plt.title('Training Accuracy Curve')
                plt.xlabel('Epochs')
                plt.ylabel('Binary or Categorical Accuracy')

                try:
                    plt.show()
                except KeyboardInterrupt:
                    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Network visualization module.')
    parser.add_argument("filename", help='Path to the file')
    parser.add_argument("--graph", "-g",  action='store_true')

    args = parser.parse_args()

    kwargs = {'graph': args.graph}

    main(args.filename, **kwargs)
