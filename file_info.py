import argparse

import numpy as np
from Bio import SeqIO

from evolutron.tools import Handle


def folder_info(foldername):
    import pandas as pd
    import glob

    h_files = glob.glob(foldername + '/*CoDERwCross.history.npz')

    opt_matrix = []

    for f in h_files:
        handle = Handle.from_filename(f)

        try:
            with np.load(f) as data:
                val_loss = data['val_loss']
                train_loss = data['train_loss']
        except KeyError:
            with np.load(f) as data:
                val_loss = data['val_acc_mem']
                train_loss = data['train_acc_mem']

        opt_matrix.append((handle.filter_size, handle.filters, train_loss[-1], val_loss[-1]))

    opt_matrix = np.asarray(opt_matrix)

    train_acc_df = pd.DataFrame(index=np.unique(opt_matrix[:, 0]).astype(np.int64),
                                columns=np.unique(opt_matrix[:, 1]).astype(np.int64)).astype(np.float32)
    val_acc_df = pd.DataFrame(index=np.unique(opt_matrix[:, 0]).astype(np.int64),
                              columns=np.unique(opt_matrix[:, 1]).astype(np.int64)).astype(np.float32)

    for f_s, f_l, tr_ac, vl_ac in opt_matrix:
        train_acc_df.loc[int(f_s), int(f_l)] = tr_ac
        val_acc_df.loc[int(f_s), int(f_l)] = vl_ac

    from pandas.tools.plotting import table
    import matplotlib.pyplot as plt
    ax1 = plt.subplot(211, frame_on=False)
    ax1.xaxis.set_visible(False)
    ax1.yaxis.set_visible(False)
    ax1.set_title('Training Accuracies')

    values = np.nan_to_num(train_acc_df.as_matrix())
    normal = plt.Normalize(values.min() - 1, values.max() + 1)
    tab1 = table(ax1, train_acc_df, loc='center', cellColours=plt.cm.YlGn(normal(values)))

    ax2 = plt.subplot(212, frame_on=False)
    ax2.xaxis.set_visible(False)
    ax2.yaxis.set_visible(False)
    ax2.set_xlabel('Number of filters')
    ax2.set_ylabel('Filter Size')
    ax2.set_title('Validation Accuracies')

    values = np.nan_to_num(val_acc_df.as_matrix())
    normal = plt.Normalize(values.min() - 2, values.max() + 2)
    tab2 = table(ax2, val_acc_df, loc='center',
                 cellLoc='center',
                 cellColours=plt.cm.YlGn(normal(values)))

    fig = plt.gcf()
    fig.set_facecolor('white')
    fig.set_size_inches(8, 11.5)
    plt.savefig('CoMET/show/' + handle.dataset + '.hyper_mat.pdf')
    plt.show()


def main(filename, **options):
    import evolutron.networks.las as nets
    from evolutron.trainers.thn import DeepTrainer

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
    parser.add_argument("--graph", action='store_true')
    parser.add_argument("--folder", action='store_true')

    args = parser.parse_args()

    kwargs = {'graph': args.graph}

    if args.folder:
        folder_info(args.filename)
    else:
        main(args.filename, **kwargs)
