import lasagne
import theano
import theano.tensor as ten

inp = ten.tensor3('input', dtype=theano.config.floatX)

input = lasagne.layers.InputLayer(input_var=inp,
                                  shape=(1, 20, None),
                                  name='Input')
conv = lasagne.layers.Conv1DLayer(input,
                                  num_filters=5,
                                  filter_size=10,
                                  flip_filters=False,
                                  nonlinearity=lasagne.nonlinearities.identity,
                                  W=lasagne.init.GlorotUniform('relu'),
                                  stride=1,
                                  name='Conv1')

c_f = theano.function([inp], conv.get_output_for(inp))

maxpool = lasagne.layers.GlobalPoolLayer(conv,
                                         pool_function=ten.max,
                                         name='MaxPool')

output = lasagne.layers.get_output(maxpool)

f = theano.function([inp], [output])


demaxpool = lasagne.layers.InverseLayer(maxpool, maxpool)

output1 = lasagne.layers.get_output(demaxpool)

f1 = theano.function([inp], [output1])

deconv = lasagne.layers.InverseLayer(demaxpool, conv)

output2 = lasagne.layers.get_output(deconv)

f2 = theano.function([inp], [output2])
