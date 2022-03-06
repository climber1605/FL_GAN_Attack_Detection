from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# TensorFlow, tf.keras and tensorflow_federated
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import functools
import glob
import os
import PIL
import time
import math
import pickle as pkl
from pathlib import Path

# tf2.0 setting
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
for dev in physical_devices:
    tf.config.experimental.set_memory_growth(dev, True)
np.random.seed(0)

# Constants
Round = 120
clients_per_round = 10
Batch_size = 2048
Gan_epoch = 1
num_class = 10
num_try = 3
dirname = 'result/MNIST/with_attack/Manhattan/conv/mid/without_dp/nonoverlap/'

ACC_THERESHOLD = 0.96
ACC_DIFF = 0.001
SELECT_ACC_THRESHOLD = 0.6
ATTACK_ACC_THRESHOLD = 0.85

neuron_ratio = 0.0005
target_layer_idx = 3
num_filter = 128
noise_scale = 0.01
with_attack = True
with_dp = False

BATCH_SIZE = 256
noise_dim = 100
num_examples_to_generate = 36
num_to_merge = 500
# num_to_merge = 50
seed = tf.random.normal([num_examples_to_generate, noise_dim])
seed_merge = tf.random.normal([num_to_merge, noise_dim])

def permutate(nums):
    for i in range(len(nums)):
        j = np.random.randint(i, len(nums))
        while nums[i] == j or nums[j] == i:
            j = np.random.randint(i, len(nums))
        nums[i], nums[j] = nums[j], nums[i]
    return nums

target_labels = permutate(list(range(clients_per_round)))
print(target_labels)

#########################################################################
##                             Load Data                               ##
#########################################################################

# Data
(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
#print(train_images.shape, train_labels.shape)
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5   # Normalization
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')
test_images = (test_images - 127.5) / 127.5   # Normalization

state = np.random.get_state()
np.random.shuffle(train_images)
np.random.set_state(state)
np.random.shuffle(train_labels)

filters = [[None for _ in range(Round)] for _ in range(clients_per_round)]
updates = [[[] for _ in range(clients_per_round)] for _ in range(Round)]
#clients = [1,5,7]
#clients = [0, 2, 3, 4, 6, 9]
#clients = [2]
clients = list(range(9, 10))

#for malicious_client in range(7, clients_per_round - 2):
for malicious_client in clients:
    target_label = target_labels[malicious_client]
    filename = ''
    if with_attack:
        if with_dp:
            filename = 'weights_distance_with_attack_{}_{}_{}_{}.txt'.format(malicious_client, target_label, num_filter, str(noise_scale).replace('.', ''))
        else:
            filename = 'weights_distance_with_attack_{}_{}_{}.txt'.format(malicious_client, target_label, num_filter)
    else:
        file_name = 'weights_distance_without_attack_{}.png'.format(num_filter)
    #predict_file = open(dirname + filename, 'w')
    avg_acc = avg_fpr = avg_fnr = 0
    for trial in range(0, 1):
        Test_accuracy = []
        Models = { }
        Client_data = {}
        Client_labels = {}
        # Each Client owns different data, Attacker has no targeted samples
        for i in range(clients_per_round):
            # Client_data.update({i:np.vstack((np.vstack((train_images[train_labels==i], train_images[train_labels==(i+1)%9])), train_images[train_labels==(i+2)%9]))})
            # Client_labels.update({i:np.append(np.append(train_labels[train_labels==i], train_labels[train_labels==(i+1)%9]), train_labels[train_labels==(i+2)%9])})
            
            # One for 5 classes
            # Client_data.update({i:train_images[train_labels==(i*5)]})
            # Client_labels.update({i:train_labels[train_labels==(i*5)]})
            # for j in range(4):
            #     Client_data[i] = np.vstack((Client_data[i], train_images[train_labels==(i+j+1)]))
            #     Client_labels[i] = np.append(Client_labels[i], train_labels[train_labels==(i+j+1)])
            
            # One for 2 classes
            #Client_data.update({i:train_images[train_labels==i]})
            #Client_labels.update({i:train_labels[train_labels==i]})
            #Client_data[i] = np.vstack((Client_data[i], train_images[train_labels==(i+1)%num_class]))
            #Client_labels[i] = np.append(Client_labels[i], train_labels[train_labels==(i+1)%num_class])
            
            # Each Client has one class
            Client_data.update({i:train_images[train_labels==i]})
            Client_labels.update({i:train_labels[train_labels==i]})
            
            # Each Client has random 5 classes
            #idx = np.random.choice(num_class, 5, replace=False)
            #print(i, idx)
            #Client_data.update({i:train_images[train_labels==idx[0]]})
            #Client_labels.update({i:train_labels[train_labels==idx[0]]})
            #for j in range(1, len(idx)):
            #    Client_data[i] = np.concatenate([Client_data[i], train_images[train_labels==idx[j]]])
            #    Client_labels[i] = np.concatenate([Client_labels[i], train_labels[train_labels==idx[j]]])
            #if i == malicious_client:
            #    for j in range(num_class):
            #        if j not in idx:
            #            target_label = j
            #            break
            
            # Shuffle
            state = np.random.get_state()
            np.random.shuffle(Client_data[i])
            np.random.set_state(state)
            np.random.shuffle(Client_labels[i])
        
        #print(target_label)
        attack_ds = np.array(Client_data[malicious_client])
        attack_l = np.array(Client_labels[malicious_client])


        #########################################################################
        ##                          Models Prepared                            ##
        #########################################################################

        # Models & malicious discriminator model
        def make_discriminator_model():
            model = keras.Sequential()
            model.add(keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
            model.add(keras.layers.LeakyReLU())
            model.add(keras.layers.Dropout(0.3))

            model.add(keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
            model.add(keras.layers.LeakyReLU())
            model.add(keras.layers.Dropout(0.3))

            model.add(keras.layers.Flatten())
            model.add(keras.layers.Dense(11))
            return model

        # Malicious generator model
        def make_generator_model():
            model = keras.Sequential()
            
            model.add(keras.layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.ReLU())

            model.add(keras.layers.Reshape((7, 7, 256)))
            assert model.output_shape == (None, 7, 7, 256)  # Batch size is not limited

            model.add(keras.layers.Conv2DTranspose(128, (4, 4), strides=(1, 1), padding='same', use_bias=False))
            assert model.output_shape == (None, 7, 7, 128)
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.ReLU())

            model.add(keras.layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', use_bias=False))
            assert model.output_shape == (None, 14, 14, 64)
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.ReLU())

            model.add(keras.layers.Conv2DTranspose(1, (4, 4), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
            assert model.output_shape == (None, 28, 28, 1)

            return model

        # Model
        # Sever‘s models
        model = make_discriminator_model()
        model.summary()
        #exit()

        # Clients' models
        for i in range(clients_per_round):
            Models.update({i:make_discriminator_model()})
            Models[i].compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

        #########################################################################
        ##                            Attack setup                             ##
        #########################################################################

        # Malicious gan
        generator = make_generator_model()
        malicious_discriminator = make_discriminator_model()


        # Cross entropy
        cross_entropy = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        #Loss of discriminator
        def discriminator_loss(real_output, fake_output, real_labels):
            real_loss = cross_entropy(real_labels, real_output)
            
            fake_result = np.zeros(len(fake_output))
            # Attack label
            for i in range(len(fake_result)):
                fake_result[i] = 10
            fake_loss = cross_entropy(fake_result, fake_output)
            total_loss = real_loss + fake_loss
            return total_loss

        # Loss of generator
        def generator_loss(fake_output):
            ideal_result = np.zeros(len(fake_output))
            # Attack label
            for i in range(len(ideal_result)):
                # The class which attacker intends to get
                ideal_result[i] = target_label
            
            return cross_entropy(ideal_result, fake_output)

        # Optimizer
        generator_optimizer = keras.optimizers.SGD(learning_rate=1e-3, decay=1e-7)
        discriminator_optimizer = keras.optimizers.SGD(learning_rate=1e-4, decay=1e-7)

        # Training step
        @tf.function
        def train_step(images, labels):
            noise = tf.random.normal([BATCH_SIZE, noise_dim])

            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                generated_images = generator(noise, training=True)
                
                # real_output is the probability of the mimic number
                real_output = malicious_discriminator(images, training=False)
                fake_output = malicious_discriminator(generated_images, training=False)
                
                gen_loss = generator_loss(fake_output)
                disc_loss = discriminator_loss(real_output, fake_output, real_labels = labels)

            gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)    # if r > 0 and Test_accuracy[r - 1] > 0.85 and attack_count == 0:
            gradients_of_discriminator = disc_tape.gradient(disc_loss, malicious_discriminator.trainable_variables)

            generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
            discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, malicious_discriminator.trainable_variables))

        # Train
        def train(dataset, labels, epochs):
            for epoch in range(epochs):
                start = time.time()
                for i in range(round(len(dataset)/BATCH_SIZE)):
                    image_batch = dataset[i*BATCH_SIZE:min(len(dataset), (i+1)*BATCH_SIZE)]
                    labels_batch = labels[i*BATCH_SIZE:min(len(dataset), (i+1)*BATCH_SIZE)]
                    train_step(image_batch, labels_batch)

                print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

            # Last epoch generate the images and merge them to the dataset
            generate_and_save_images(generator, epochs, seed)

        # Generate images to check the effect
        def generate_and_save_images(model, epoch, test_input):
            predictions = model(test_input, training=False)

            fig = plt.figure(figsize=(6,6))

            for i in range(predictions.shape[0]):
                plt.subplot(6, 6, i+1)
                plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
                plt.axis('off')

            plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
            plt.close()

        def cosine_distance(a, b):
            if a.shape != b.shape:
                raise RuntimeError("array {} shape not match {}".format(a.shape, b.shape))
            if a.ndim==1:
                a_norm = np.linalg.norm(a)
                b_norm = np.linalg.norm(b)
            elif a.ndim==2:
                a_norm = np.linalg.norm(a, axis=1, keepdims=True)
                b_norm = np.linalg.norm(b, axis=1, keepdims=True)
            else:
                raise RuntimeError("array dimensions {} not right".format(a.ndim))
            similiarity = np.dot(a, b.T)/(a_norm * b_norm) 
            dist = 1. - similiarity
            return dist

        #########################################################################
        ##                         Federated Learning                          ##
        #########################################################################

        # Training Preparation
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])
        model.summary()

        # model.fit(warm_up_data, warm_up_labels, validation_split=0, epochs=25, batch_size = 256)
        #del train_images, train_labels

        tmp_weight = model.get_weights()
        attack_count = 0

        layers = ['conv2d']
        activations = [[[[] for _ in range(len(layers))] for _ in range(10)] for _ in range(Round)]
        diff = [[[0 for _ in range(len(layers))] for _ in range(10)] for _ in range(Round)]
        pos = [[set() for _ in range(len(layers))] for _ in range(10)]
        diff_class = [[] for _ in range(10)]
        selected_pos = [set() for _ in range(10)]
        selected = False
        weights = [[] for _ in range(clients_per_round)]
        diff_weight = [[] for _ in range(clients_per_round)]
        weights_with_class = [[[] for _ in range(num_class)] for _ in range(clients_per_round)]
        diff_weights_with_class = [[[] for _ in range(num_class + 1)] for _ in range(clients_per_round)]
        history_acc = [[] for _ in range(clients_per_round)]
        history_loss = [[] for _ in range(clients_per_round)]

        # Federated learning
        for r in range(Round):
            print('round:'+str(r+1))
            model_weights_sum = []
            weight1= [[] for _ in range(clients_per_round)]

            for i in range(clients_per_round):
                # train the clients individually
                Models[i].set_weights(tmp_weight)
                
                train_ds = Client_data[i]
                train_l = Client_labels[i]

                # Attack (suppose client 0 is malicious)
                if with_attack and i == malicious_client and r > 0 and Test_accuracy[r - 1] > ATTACK_ACC_THRESHOLD:
                #if with_attack and i == malicious_client:
                    print("Attack round: {}".format(attack_count+1))

                    malicious_discriminator.set_weights(Models[i].get_weights())
                    # train(attack_ds, attack_l, Gan_epoch)
                    train(attack_ds, attack_l, Gan_epoch)
                    

                    predictions = generator(seed_merge, training=False)
                    malicious_images = np.array(predictions)
                    malicious_labels = np.array([10]*len(malicious_images))

                    # Merge the malicious images
                    if attack_count == 0:
                        Client_data[i] = np.vstack((Client_data[i], malicious_images))
                        # Label the malicious images
                        Client_labels[i] = np.append(Client_labels[i], malicious_labels)  
                    else:
                        Client_data[i][len(Client_data[i])-len(malicious_images):len(Client_data[i])] = malicious_images

                    attack_count += 1

                orig_conv_weight = Models[i].layers[target_layer_idx].get_weights()[0]
                orig_conv_bias = Models[i].layers[target_layer_idx].get_weights()[1]
                temp = Models[i].fit(train_ds, train_l, validation_split=0, epochs=1, batch_size = Batch_size)     
                history_acc[i].append(temp.history['accuracy'])
                history_loss[i].append(temp.history['loss'])

                if i == 0:
                    model_weights_sum = np.array(Models[i].get_weights(), dtype=object)
                else:
                    model_weights_sum += np.array(Models[i].get_weights(), dtype=object)
                
                # add laplace noise for every parameter update
                if with_dp:
                    for j in range(len(model_weights_sum)):
                        model_weights_sum[j] += np.random.laplace(scale=noise_scale, size=model_weights_sum[j].shape)

            # averaging the weights
            mean_weight = np.true_divide(model_weights_sum,clients_per_round)
            tmp_weight = mean_weight.tolist()
            del model_weights_sum


            # calculate distance between models
            # dist = [[0 for _ in range(clients_per_round)] for _ in range(clients_per_round)]
            # dist_sum = [0 for _ in range(clients_per_round)]
            # for i in range(clients_per_round):
            #     w1 = Models[i].get_weights()
            #     for j in range(i + 1, clients_per_round):
            #         w2 = Models[j].get_weights()
            #         d = 0
            #         for k in range(len(w1)):
            #             d += np.linalg.norm(w1[k] - w2[k])
            #         dist[i][j] = d
            #         dist[j][i] = dist[i][j]
            #         #print('Distance between model {} and model {}: {}'.format(i, j, dist[i][j]))
            #     for j in range(clients_per_round):
            #         dist_sum[i] += dist[i][j]
            #     print('Sum of distance between model {} and other models: {}'.format(i, dist_sum[i]))
            # idxs = list(range(clients_per_round))
            # idxs.sort(key=lambda i: dist_sum[i])
            # print(idxs)


            #if not selected and r > 0 and Test_accuracy[r - 1] > SELECT_ACC_THRESHOLD:
            if False and not selected:
                #selected_pos = [set() for _ in range(10)]
                for i in range(clients_per_round):   
                    channels = int(orig_conv_weight.shape[-1])
                    dist = [None] * channels
                    for j in range(channels, channels):
                        #w1 = np.array(orig_conv_weight[:, :, :, c]).flatten()
                        #w2 = np.array(Models[i].layers[target_layer_idx].get_weights()[0][:, :, :, c]).flatten()
                        #dist[c] = (np.linalg.norm(w2 - w1), c)
                        #dist[c] = (np.sum(abs(w2 - w1)), c)
                        #dist[c] = (np.sum(w2 - w1), c)
                        #dist[c] = (np.sum(w2 - w1) / np.sum(w1), c)
                        #dist[c] = (cosine_distance(w1.flatten(), w2.flatten()), c)
                        #dist[c] = (np.linalg.norm(w2 - w1) / (np.linalg.norm(w1) * np.linalg.norm(w2)), c)
                        w1 = orig_conv_bias[j]
                        w2 = Models[i].layers[target_layer_idx].get_weights()[1][j]
                        diff = abs(w2 - w1) / abs(w1)
                        dist[j] = (diff, j)
                    dist.sort(reverse=True)
                    for j in range(num_filter):
                        selected_pos[i].add(dist[j][1])
                    selected_pos[i] = sorted(selected_pos[i])
                    filters[i][r] = selected_pos[i][:]
                    print(i, selected_pos[i])
                selected = True
                    
            # calculate weights change in interested channels corresponding to each class and each client
            if True or selected:
                for i in range(clients_per_round):    
                    #weight1 = orig_conv_weight[:, :, :, selected_pos[i]].flatten()
                    #weight2 = Models[i].layers[target_layer_idx].get_weights()[0][:, :, :, selected_pos[i]].flatten()           
                    #diff = np.linalg.norm(weight2 - weight1)  
                    #diff = np.sum(abs(weight2 - weight1))   
                    #diff = np.sum(weight2 - weight1)
                    #diff = np.sum(weight2 - weight1) / np.sum(weight1)
                    #for j in range(int(orig_conv_weight.shape[-1])):
                    for j in range(0, 0):
                        #w1 = orig_conv_weight[:,:,:,j].flatten()
                        #w2 = Models[i].layers[target_layer_idx].get_weights()[0][:,:,:,j].flatten()           
                        #diff = cosine_distance(weight1.flatten(), weight2.flatten())
                        #diff = sum(abs((weight2 - weight1) / weight1)) / len(weight1)
                        w1 = orig_conv_bias[j]
                        w2 = Models[i].layers[target_layer_idx].get_weights()[1][j]
                        diff = abs(w2 - w1) / abs(w1)
                        updates[r][i].append(diff)
                    #w1 = orig_conv_weight[:, :, :, :].flatten()
                    #w2 = Models[i].layers[target_layer_idx].get_weights()[0][:, :, :, :].flatten() 
                    w1 = np.array(orig_conv_bias)
                    w2 = np.array(Models[i].layers[target_layer_idx].get_weights()[1])          
                    #diff = cosine_distance(weight1.flatten(), weight2.flatten())
                    #diff = np.linalg.norm(weight2 - weight1)  
                    #diff = np.linalg.norm(w2 - w1) / np.linalg.norm(w2)
                    diff = sum(abs(w2 - w1)) / sum(abs(w2))
                    #diff = sum(w2 - w1) / sum(w2)
                    #print(i, diff) 
                    diff_weight[i].append(diff)

            # evaluate
            model.set_weights(tmp_weight)
            test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
            Test_accuracy.append(test_acc)
            print('\nTest accuracy:', test_acc, 'Test loss:', test_loss)

        # draw weights change corresponding to each client
        colors = ['r--', 'g--', 'b--', 'c--', 'm--', 'r', 'g', 'b', 'c', 'm']
        plt.cla()
        for c in range(clients_per_round):
            plt.plot(diff_weight[c], colors[c], label='client {}'.format(c))
        if with_attack:
            plt.title('weights change(Malicious client: {}, target label: {})'.format(malicious_client, target_label))
        else:
            plt.title('weights change(No malicious client)')
        plt.xlabel('round')
        plt.ylabel('weights distance')
        plt.legend()
        pic_name = ''
        if with_attack:
            if with_dp:
                pic_name = 'weights_distance_with_attack_{}_{}_{}_{}_{}.png'.format(malicious_client, target_label, num_filter, str(noise_scale).replace('.', ''), trial)
            else:
                pic_name = 'weights_distance_with_attack_{}_{}_{}_{}.png'.format(malicious_client, target_label, num_filter, trial)
        else:
            pic_name = 'weights_distance_without_attack_{}_{}.png'.format(num_filter, trial)
        #plt.savefig(pic_name)
        #plt.show()
        plt.close()

        # # 绘制训练 & 验证的准确率值
        # selected_clients = [3, 0, 2, 4, 6, 8]
        # colors = ['r--', 'g--', 'b--', 'r', 'g', 'b']
        # for i, client in enumerate(selected_clients):
        #     plt.plot(history_acc[client], colors[i], label='client{}'.format(client))

        # plt.title('Model accuracy(Malicious client: {}, target label: {})'.format(malicious_client, target_label))
        # plt.ylabel('Accuracy')
        # plt.xlabel('Epoch')
        # plt.legend()
        # if with_attack:
        #     plt.savefig('training_accuracy_with_attack_{}_1.png'.format(target_label))
        # else:
        #     plt.savefig('training_accuracy_without_attack.png')
        #plt.show()

        # # 绘制训练 & 验证的损失值
        for i in range(clients_per_round):
            plt.plot(history_loss[i], colors[i], label='client{}'.format(i))
        if with_attack:
            plt.title('Model loss(Malicious client: {}, target label: {})'.format(malicious_client, target_label))
        else:
            plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        filename = pic_name.replace('weights_distance', 'training_loss')
        #plt.show()
        #plt.savefig(filename)

        #print(diff_weight)
        #print(malicious_client, target_label)
        #predict_file.write('malicious client: {}, target label: {}\n'.format(malicious_client, target_label))
        #filename = pic_name.replace('png', 'pkl')
        #with open(dirname + filename, 'wb') as f:
        #    pkl.dump(updates, f)
        
        filename = pic_name.replace('.png', '.pkl')
        if not Path(filename).is_file():
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            print(f'dump file {dirname + filename}')
            with open(dirname + filename, 'wb') as f:
                pkl.dump(diff_weight, f)
                #pass

        # save filters' change
        #filename = pic_name.replace('png', 'pkl').replace('weights_distance', 'filters_change')
        #with open(dirname + filename, 'wb') as f:
        #    pkl.dump(filters, f)     
        #for i in range(len(filters)):
        #    for j in range(len(filters[i])):
        #        print(i, j, filters[i][j])

    #     # time series outlier detection based on sliding window prediction
    #     k = 10
    #     t = 2.539
    #     #t = 3.106
    #     suspects = []
    #     data = diff_weight
    #     TP = TN = FP = FN = 0

    #     for l in range(clients_per_round):
    #         predicted = []
    #         low = []
    #         up = []
    #         for i in range(len(data[l])):
    #             s = w = 0
    #             minj, maxj = max(0, i - k), min(len(data[l]), i + k + 1)
    #             for j in range(minj, i):
    #                 s += (j - i + k + 1) * data[l][j]
    #                 w += j - i + k + 1
    #             for j in range(i + 1, maxj):
    #                 s += (i + k - j + 1) * data[l][j]
    #                 w += i + k - j + 1
    #             s /= w
    #             predicted.append(s)
    #             temp = np.array(data[l][minj: maxj])
    #             r = t * math.sqrt(np.var(temp)) * math.sqrt(1 + 1 / (2 * k))
    #             #print(s,r,s+r,s-r)
    #             up.append(s + r)
    #             low.append(s - r)

    #         plt.cla()
    #         plt.plot(data[l], 'b', label='real')
    #         plt.plot(predicted, 'r', label='predicted')
    #         plt.plot(up, 'b--', label='predicted upper')
    #         plt.plot(low, 'r--', label='predicted lower')
    #         plt.legend()
    #         #plt.show()
    #         plt.close()

    #         for i in range(len(data[l])):
    #             if not low[i] <= data[l][i] <= up[i]:
    #                 print('Found outlier')
    #                 print(l, i, low[i], data[l][i], up[i])
    #                 suspects.append(l)
    #                 break

    #     if len(suspects):
    #         print('Found {} suspects'.format(len(suspects)))
    #         print(suspects)
    #     else:
    #         print('Not found')

    #     predict_file.write('trial #{}, found {} suspects: {}\n'.format(trial, len(suspects), ' '.join(map(str, suspects))))
    #     for c in range(clients_per_round):
    #         if c in suspects:
    #             if c == malicious_client:
    #                 TP += 1
    #             else:
    #                 FP += 1
    #         else:
    #             if c == malicious_client:
    #                 FN += 1
    #             else:
    #                 TN += 1
    #     accuracy = (TP + TN) / (TP + TN + FP + FN)
    #     false_postive_rate = FP / (FP + TN)
    #     false_negative_rate = FN / (FN + TP)
    #     predict_file.write('accuracy: {}, false postive rate: {}, false negative rate: {}\n'.format(accuracy, false_postive_rate, false_negative_rate))

    #     avg_acc += accuracy
    #     avg_fpr += false_postive_rate
    #     avg_fnr += false_negative_rate
    
    # avg_acc /= num_try
    # avg_fpr /= num_try
    # avg_fnr /= num_try
    # predict_file.write('average accuracy: {}, average false postive rate: {}, average false negative rate: {}\n'.format(avg_acc, avg_fpr, avg_fnr))
    # predict_file.close()
