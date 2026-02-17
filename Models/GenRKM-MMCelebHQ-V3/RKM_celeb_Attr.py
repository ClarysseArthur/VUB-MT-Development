from utils import *

# Load Training Data ===============================================================================
transform = transforms.Compose([transforms.CenterCrop(128), transforms.ToTensor()])

trainset = datasets.ImageFolder(root='../dataset/', transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=N, shuffle=None)

dataiter = iter(trainloader)
xtrain, _ = dataiter.next()
ytrain = np.load('../dataset/attr.npy')  # labels with one-hot encoding
ytrain = torch.from_numpy(ytrain[:N, :]).float()

classes = ('_o_Clock_Shadow','Arched_Eyebrows','Attractive','Bags_Under_Eyes',
           'Bald','Bangs','Big_Lips','Big_Nose','Black_Hair','Blond_Hair','Blurry',
           'Brown_Hair','Bushy_Eyebrows','Chubby','Double_Chin','Eyeglasses','Goatee',
           'Gray_Hair','Heavy_Makeup','High_Cheekbones','Male','Mouth_Slightly_Open',
           'Mustache','Narrow_Eyes','No_Beard','Oval_Face','Pale_Skin','Pointy_Nose',
           'Receding_Hairline','Rosy_Cheeks','Sideburns','Smiling ','Straight_Hair','Wavy_Hair',
           'Wearing_Earrings','Wearing_Hat','Wearing_Lipstick','Wearing_Necklace','Wearing_Necktie','Young')

# visualize train images ===========
# gt = plt.figure(2)
# for j in range(9):
#     plt.subplot(3, 3, j + 1)
#     plt.imshow(convert_to_imshow_format(xtrain[j, :, :, :]))
# gt.suptitle('Ground Truth')

ct = time.strftime("%Y%m%d-%H%M")
dirs = create_dirs(ct=ct)
dirs.create()

def kPCA(X, Y):
    nh1 = output1.size(0)
    a = torch.mm(X, torch.t(X)) + torch.mm(Y, torch.t(Y))
    oneN = torch.div(torch.ones(nh1, nh1), nh1).to(device)
    a = a - torch.mm(oneN, a) - torch.mm(a, oneN) + torch.mm(torch.mm(oneN, a), oneN)   # centering
    h, s, v = torch.svd(a, some=False)
    return h[:, : h_dim], s


# Energy function
def my_loss(output1, X, output2, Y):
    h, s = kPCA(output1, output2)
    U = torch.mm(torch.t(output1), h)
    V = torch.mm(torch.t(output2), h)

    x_tilde = net3(torch.mm(h, torch.t(U)))
    y_tilde = net4(torch.mm(h, torch.t(V)))

    # cost
    f1 = torch.trace(torch.mm(torch.mm(output1, U), torch.t(h))) + torch.trace(torch.mm(torch.mm(output2, V), torch.t(h)))
    f2 = 0.5 * torch.trace(torch.mm(h, torch.mm(torch.diag(s[:h_dim]), torch.t(h))))
    f3 = 0.5 * ((torch.trace(torch.mm(torch.t(U), U))) + (torch.trace(torch.mm(torch.t(V), V))))
    recon_loss2 = torch.nn.MSELoss()
    recon_loss1 = torch.nn.MSELoss()
    f4 = recon_loss1(x_tilde.view(-1, 49152), X.view(-1, 49152)) + recon_loss2(y_tilde.view(-1, 40), Y.view(-1, 40))  # reconstruction loss

    loss = - f1 + f3 + f2 + 0.5 * (- f1 + f3 + f2) ** 2 + 100 * f4
    return loss


params = list(net1.parameters()) + list(net3.parameters()) + list(net2.parameters()) + list(net4.parameters())
optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=0)

# Train ===============================================================================
l_cost = 6  # Costs from where checkpoints will be saved
t = 1
cost = np.inf  # Initialize cost
while cost > 0.2 and t <= max_epochs:  # run epochs until convergence
    permutation = torch.randperm(xtrain.size()[0])
    avg_loss = 0
    for i in range(0, N, mb_size):
        indices = permutation[i:i + mb_size]
        datax = xtrain[indices, :, :, :].to(device)
        datay = ytrain[indices, :].to(device)
        output1, output2 = net1(datax), net2(datay)
        loss = my_loss(output1, datax, output2, datay)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss += loss.detach().cpu().numpy()
    cost = avg_loss

    # Remember lowest cost and save checkpoint
    is_best = cost < l_cost
    l_cost = min(cost, l_cost)
    dirs.save_checkpoint({
        'epochs': t + 1,
        'net1_state_dict': net1.state_dict(),
        'net3_state_dict': net3.state_dict(),
        'net2_state_dict': net2.state_dict(),
        'net4_state_dict': net4.state_dict(),
        'l_cost': l_cost,
        'optimizer': optimizer.state_dict()}, is_best)

    print(t, cost)
    t += 1
print('Finished Training. Lowest cost: {}'.format(l_cost))

if os.path.exists('cp/{}'.format(dirs.dircp)):
    sd_mdl = torch.load('cp/{}'.format(dirs.dircp))
    net1.load_state_dict(sd_mdl['net1_state_dict'])
    net3.load_state_dict(sd_mdl['net3_state_dict'])
    net2.load_state_dict(sd_mdl['net2_state_dict'])
    net4.load_state_dict(sd_mdl['net4_state_dict'])

output1 = net1(xtrain.to(device))
output2 = net2(ytrain.to(device))
h, s = kPCA(output1, output2)

U = torch.mm(torch.t(output1), h)
V = torch.mm(torch.t(output2), h)

# Save Model ============
torch.save({'net1': net1,
            'net3': net3,
            'net2': net2,
            'net4': net4,
            'net1_state_dict': net1.state_dict(),
            'net3_state_dict': net3.state_dict(),
            'net2_state_dict': net2.state_dict(),
            'net4_state_dict': net4.state_dict(),
            'xtrain': xtrain,
            'ytrain': ytrain,
            'classes': classes,
            'h': h, 'N': N, 's': s, 'U': U, 'V': V}, 'out/{}'.format(dirs.dirout))
