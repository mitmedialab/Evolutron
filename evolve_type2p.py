#!/usr/bin/env python

from __future__ import print_function

import lasagne
import numpy as np
import theano
import theano.tensor as ten

from evolutron.tools import num2aa, nt2prob, aa2num, num2hot
from evolutron.trainers.thn import ConvType2p

# def main(mode, padded, handle=None, **options):
padded = True

filename = 'networks/type2p/200_150_50_50.model.npz'
filters = int(filename.split('_')[2])
filter_size = int(filename.split('_')[2])

# Test sequences
ecori = 'MSNKKQSNRLTEQHKL' \
        'SQGVIGIFGDYAKAHDLAVGEVSKLVKKALSNEYPQLSFRYRDSIKKTEINEALKKIDPDLGGTLFVSNSSIKPDGGIVEVKDDYGEWRVVLVA' \
        'EAKHQGKDIINIRNGLLVGKRGDQDLMAAGNAIERSHKNISEIANFMLSESHFPYVLFLEGSNFLTENISITRPDGRVVNLEYNSGILNRLDRLT' \
        'AANYGMPINSNLCINKFVNHKDKSIMLQAASIYTQGDGREWDSKIMFEIMFDISTTSLRVLGRDLFEQLTSK'

ecori_pdb = 'SQGVIGIFGDYAKAHDLAVGEVSKLVKKALSNEYPQLSFRYRDSIKKTEINEALKKIDPDLGGTLFVSNSSIKPDGGIVEVKDDYGEWRVVLVA' \
            'EAKHQGKDIINIRNGLLVGKRGDQDLMAAGNAIERSHKNISEIANFMLSESHFPYVLFLEGSNFLTENISITRPDGRVVNLEYNSGILNRLDRLT' \
            'AANYGMPINSNLCINKFVNHKDKSIMLQAASIYTQGDGREWDSKIMFEIMFDISTTSLRVLGRDLFEQLTSK'

muni = 'MGKSE' \
       'LSGRLNWQALAGLKASGAEQNLYNVFNAVFEGTKYVLYEKPKHLKNLYAQVVLPDDVIKEIFNPLIDLSTTQWGVSPDFAIENTETHKILFGEIKR' \
       'QDGWVEGKDPSAGRGNAHERSCKLFTPGLLKAYRTIGGINDEEILPFWVVFEGDITRDPKRVREITFWYDHYQDNYFMWRPNESGEKLVQHFNEKLKKYLD'

muni_pdb = 'LSGRLNWQALAGLKASGAEQNLYNVFNAVFEGTKYVLYEKPKHLKNLYAQVVLPDDVIKEIFNPLIDLSTTQWGVSPAFAIENTETHKILFGEIKR' \
           'QDGWVEGKDPSAGRGNAHERSCKLFTPGLLKAYRTIGGINDEEILPFWVVFEGDITRDPKRVREITFWYDHYQDNYFMWRPNESGEKLVQHFNEKLKKYLD'


muni_20000 = 'MGKSE' \
        'LSGRLNWQALAGLKASGAEQNLYNVFNAVFEGTKYVLYEKPKHLKNLYAQVVLPDDVIKEIFNPLIDLSTTQWGVSPDFAIENTETHKILFGEIKR' \
        'QDGWVEGKDPSAGRGNAHERSCKLFTPGLLKAYRTIGGINDEEILPFWVVFEGDITRDPKRVREITFWYDHYQDNYFMWRPNESGEKLVQHFNEKLKKYLD'

muni_pdb_20000 ='LSGRLNWQALAGLKASGAEQNLYNVFNAVFEGTKYVLYEKPKHLKNLYAQVVLPDDVIKEIFNPLIDLSTTQWGVSPAFAIENTETHKILFGEIKR' \
                'QDGWVEGKDPSAGRGNAHERSCKLFTPGLLKAYRTIGGINDEEILPFWVVFEGDITRDPKRVREITFWYDHYQDNYFMWRPNESGEKLVQHFNEKLKKYLD'

noise = np.zeros((1, 20, 300))
test = num2hot(aa2num(muni_pdb))
output = nt2prob('GAATTC')

net_arch = ConvType2p(pad_size=None, filters=filters, filter_size=filter_size)

with np.load(filename) as f:
    param_values = [f['arr_%d' % i] for i in range(len(f.files))]

lasagne.layers.set_all_param_values(net_arch.network, param_values)


# inps = theano.shared(noise.astype(theano.config.floatX), name='xdata')
inps = theano.shared(test, name='xdata')
inp = net_arch.inp

loss, acc, prediction = net_arch.build_loss(deterministic=True)

gparams = ten.grad(loss, wrt=inp)
updates = [(inps, inps - .1 * gparams)]
# updates = lasagne.updates.nesterov_momentum(loss, net_arch.network.input_layers[0].input,
#                                             learning_rate=.01,
#                                             momentum=0.975)

# Compile a function performing a training step on a mini-batch
back_fn = theano.function(inputs=[inp, net_arch.targets],
                          outputs=[loss, prediction],
                          updates=updates,
                          allow_input_downcast=True)

# network_inv =


for i in range(1, 20000):
    l, pred = back_fn(inps.get_value(), output)

    if i % 1000 == 0:
        print(l)
        print(pred)

curr = inps.get_value(borrow=True)

result = num2aa(np.argmax(curr.squeeze(),0))
print(result)
print(muni)

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Script to use Evolutron to predict Type2 Restriciton Enzymes binding '
#                                                  'motifs\n'
#                                                  'Input: Type2(p) aa sequences and their corresponding '
#                                                  'recognition sites\n'
#                                                  'from Microarray or Bacterial one hybrid experiments')
#
#     parser.add_argument("mode", choices=['k_fold', 'train'],
#                         help='Choose mode of operation.')
#     parser.add_argument("filters", type=int,
#                         help='Number of filters in the convolutional layers.')
#     parser.add_argument("filter_size", type=int,
#                         help='Size of filters in the convolutional layer.')
#
#     parser.add_argument("-e", "--epochs", default=200, type=int,
#                         help='Number of training epochs (default: 200).')
#     parser.add_argument("-b", "--batch_size", type=int, default=1,
#                         help='Size of minibatch. If not padded it defaults to 1, else to 150.')
#     parser.add_argument("-f", "--folds", default=10, type=int,
#                         help='Number of folds when in KFold mode.')
#     parser.add_argument("--padded", action='store_true',
#                         help='Toggle to pad protein sequences.')
#
#     parser.add_argument("-i", "--handle", default=None)
#
#     args = parser.parse_args()
#
#     if args.padded and args.batch_size == 1:
#         args.batch_size = 150
#
#     kwargs = {'mode': args.mode,
#               'num_epochs': args.epochs,
#               'batch_size': args.batch_size,
#               'filters': args.filters,
#               'filter_size': args.filter_size,
#               'padded': args.padded,
#               'handle': args.handle}
#     main(**kwargs)