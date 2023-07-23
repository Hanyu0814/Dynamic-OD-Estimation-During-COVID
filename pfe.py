import torch
import numpy as np
import math

def get_gradient(A, y, x):
    (n ,d) = A.size()
    # tmp = y - torch.mm(A, x.view(d, -1))
    tmp = y.view(n, -1) - torch.mm(A, x.view(d, -1))
    return - torch.mm(A.t(), tmp.view(n, -1)) / np.float(n)

def get_loss(A, y, x):
    (n ,d) = A.size()
    return torch.sum(torch.pow(y.view(n, -1) - torch.mm(A, x.view(d, -1)), 2))

def nnls(A, y, num_epoch, batch_size, learning_rate, adagrad = False, 
             use_GPU = False, D_vec = None, D_vec_weight = 0.1, verbose = False):
    (n, d) = A.shape
    f = open("Output_nnls_"+str(num_epoch)+".txt", "a")
    f.write("Epoch_number, Avg_loss, SQRT_of_avg_loss,L2_loss, L2_norm_of_true_volume, Normalized_L2_loss")
    scale = np.sqrt(6) / np.sqrt(np.float(d))
    x = (np.random.rand(d) * 2 - 1.0) * scale
    # A_torch = torch.FloatTensor(A)
    y_torch = torch.FloatTensor(y)
    x_torch = torch.FloatTensor(x)
    if D_vec is not None:
        D_vec_torch = torch.FloatTensor(D_vec)
    if use_GPU:
        # A_torch = A_torch.cuda()
        y_torch = y_torch.cuda()
        x_torch = x_torch.cuda()
        if D_vec is not None:
            D_vec_torch = D_vec_torch.cuda()

    counter = 0  # !! My code
    counter2 = 0  # !! My code
    lr_loss_check = [float('inf')] # !! My code

    for epoch in range(num_epoch):
        print(epoch)
        if adagrad:
            sum_g_square = 0
        # if verbose: 
        #     print "Epoch:", epoch
        #     loss = get_loss(A_torch, y_torch, x_torch) 
        #     print "Current loss is:", loss
        seq = np.random.permutation(n)
        # train_sample_list = np.array_split(seq, len(seq) / batch_size)
        if int(len(seq) / batch_size) == 0: # My added if line
            # print "Number of batch sizes larger than the number of instances." # My added if line
            train_sample_list = np.array_split(seq, 1) # My added if line
        else:
            # print "Number of batches: ", len(seq) / batch_size
            # Divide the permuted indices in the number of batches # His code
            train_sample_list = np.array_split(seq, len(seq) / batch_size) # His code

        lr_loss_check_temp = []
        for sample_ind in train_sample_list:
            # if use_GPU:
            #     sample_ind_torch = torch.cuda.LongTensor(sample_ind)
            # else:
            #     sample_ind_torch = torch.LongTensor(sample_ind)
            # A_sub = torch.index_select(A_torch, 0, sample_ind_torch)
            A_sub = torch.FloatTensor(A[sample_ind, :])
            if use_GPU:
                A_sub = A_sub.cuda()
            if use_GPU:
                sample_ind_torch = torch.cuda.LongTensor(sample_ind)
            else:
                sample_ind_torch = torch.LongTensor(sample_ind)
            y_sub = y_torch[sample_ind_torch]
            # g = get_gradient(A_sub, y_torch, x_torch)
            g = get_gradient(A_sub, y_sub, x_torch)
            # print("type ", type(g))
            # print("shape", g.shape)

            # print("type ", type(x_torch))
            # print("shape", x_torch.shape)
            if D_vec is not None:
                g += D_vec_weight * D_vec_torch.view(d, -1)
            if adagrad:
                sum_g_square = sum_g_square + torch.pow(torch.flatten(g), 2)
                # x_torch = x_torch - learning_rate * g / torch.sqrt(sum_g_square) #his code
                x_torch = x_torch - learning_rate * torch.flatten(g) / (torch.sqrt(sum_g_square)+1e-8) #ali's code
#                 print sum_g_square
            else:
                # x_torch = x_torch - learning_rate / np.float(epoch + 1) * g
                x_torch = x_torch - learning_rate * torch.flatten(g) / np.float(epoch + 1) # my code
            x_torch[x_torch != x_torch] = 0.0
            x_torch = torch.clamp(x_torch, min = 0.0)
            # loss = get_loss(A, y, x) # !! His code
            loss = get_loss(A_sub, y_sub, x_torch) # !! ali code
            lr_loss_check_temp.append(loss.cpu().item())
            avg_loss_temp = sum(lr_loss_check_temp)/float(n)
            normalized_loss_temp = math.sqrt(avg_loss_temp)/np.linalg.norm(y)
            min_temp = min(lr_loss_check)

            lr_loss_check.append(avg_loss_temp)
            
            f.write("\n %i, %.2f, %.2f,%.2f, %.2f, %.2f" % (epoch, avg_loss_temp,math.sqrt(avg_loss_temp), math.sqrt(sum(lr_loss_check_temp)), math.sqrt(sum(np.power(y, 2))), (math.sqrt(sum(lr_loss_check_temp)))/math.sqrt(sum(np.power(y, 2)))))


            # print("\nEpoch number: %i, Avg loss: %.2f, SQRT of avg loss: %.2f" % (epoch, avg_loss_temp, math.sqrt(avg_loss_temp)))
            # print("L2 loss:  %.2f, L2 norm of true volume:  %.2f, Normalized L2 loss:  %.2f" % (math.sqrt(sum(lr_loss_check_temp)), math.sqrt(sum(np.power(y, 2))), (math.sqrt(sum(lr_loss_check_temp)))/math.sqrt(sum(np.power(y, 2)))))

    if use_GPU:
        x_torch = x_torch.cpu()
    return (x_torch.numpy(), loss)



def new_nnls(A, y, maxiter, batch_size, step_size, adagrad = False, verbose = False):
    # L.dot(A)
    (n,d) = A.shape
    # x = np.random.rand(d)
    scale = np.sqrt(6) / np.sqrt(np.float(d))
    x = (np.random.rand(d) * 2 - 1.0) * scale
    for epoch in range(maxiter):
        if adagrad:
                sum_g_square = 1e-6
        seq = np.random.permutation(n)
        train_sample_list = np.array_split(seq, len(seq) / batch_size)
        for sample_ind in train_sample_list:
            A_sub = A[sample_ind, :]
            y_sub = y[sample_ind]
    #         print y_sub.shape
            g = - A_sub.T.dot(y_sub - A_sub.dot(x)) / batch_size
    #         print g
            if adagrad:
                sum_g_square = sum_g_square + np.power(g, 2)
    #             print sum_g_square
                x = x - step_size * g / np.sqrt(sum_g_square)
            else:
                x = x - step_size / np.float(epoch + 1) * g
        x = np.maximum(0, x)
        loss = np.linalg.norm(y - A.dot(x))
        if verbose:
            print(epoch, loss)
    return (x, loss)