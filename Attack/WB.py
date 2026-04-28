import torch


class WB_Attack:
    def __init__(self, net_in1, net_out1, recon_loss1, net_in2=None, net_out2=None, recon_loss2=None, net_in3=None, net_out3=None, recon_loss3=None, U=None, V=None, W=None, kPCA=None, vta=[True, True], opt=None):
        self.net_in1 = net_in1
        self.net_in2 = net_in2
        self.net_in3 = net_in3

        self.net_out1 = net_out1
        self.net_out2 = net_out2
        self.net_out3 = net_out3

        self.kPCA = kPCA
        self.opt = opt
        self.vta = vta
        self.U = U
        self.V = V
        self.W = W

        self.net_in1.zero_grad()
        self.net_in2.zero_grad() if self.net_in2 is not None else None
        self.net_in3.zero_grad() if self.net_in3 is not None else None
        self.net_out1.zero_grad()
        self.net_out2.zero_grad() if self.net_out2 is not None else None
        self.net_out3.zero_grad() if self.net_out3 is not None else None

        self.recon_loss1 = recon_loss1
        self.recon_loss2 = recon_loss2
        self.recon_loss3 = recon_loss3

    def _kPCA(self, x_en, y_en=None, z_en=None):
        if y_en is not None and z_en is not None:
            return self.kPCA(x_en, y_en, z_en, 3)
        elif y_en is not None:
            return self.kPCA(x_en, y_en, None, 2)
        else:
            return self.kPCA(x_en, None, None, 1)

    def run_model_batch(self, b1, b2=None, b3=None):
        x_en = self.net_in1(b1)
        y_en = self.net_in2(b2) if b2 is not None else None
        z_en = self.net_in3(b3) if b3 is not None else None

        h, _ = self._kPCA(x_en, y_en, z_en)
        U = torch.mm(x_en.t(), h)
        V = torch.mm(y_en.t(), h) if y_en is not None else None
        W = torch.mm(z_en.t(), h) if z_en is not None else None

        x_tilde = self.net_out1(torch.mm(h, U.t()))
        y_tilde = self.net_out2(torch.mm(h, V.t())) if V is not None else None
        z_tilde = self.net_out3(torch.mm(h, W.t())) if W is not None else None
        return x_tilde, y_tilde, z_tilde

    def create_adversarial_batch_fgsm(self, in1, in2=None, in3=None, epsilon=0.1): # Source: https://www.tensorflow.org/tutorials/generative/adversarial_fgsm
        if self.vta[0] and self.vta[1]:
            x = in1.clone().detach().to(self.opt.device)
            y = in2.clone().detach().to(self.opt.device) if in2 is not None else None

            x.requires_grad_(True)
            y.requires_grad_(True) if y is not None else None

            z = in3.to(self.opt.device) if in3 is not None else None

        elif self.vta[0]:
            x = in1.clone().detach().to(self.opt.device)
            x.requires_grad_(True)

            y = in2.to(self.opt.device) if in2 is not None else None
            z = in3.to(self.opt.device) if in3 is not None else None

        elif self.vta[1]:
            y = in2.clone().detach().to(self.opt.device) if in2 is not None else None
            y.requires_grad_(True) if y is not None else None

            x = in1.to(self.opt.device)
            z = in3.to(self.opt.device) if in3 is not None else None

        out_x, out_y, out_z = self.run_model_batch(x, y, z)             # 1. Forward clean image batch through the model
        
        if in3 is not None and in2 is not None:
            loss = self.recon_loss1(out_x, in1) + self.recon_loss2(out_y, in2) + self.recon_loss3(out_z, in3) # 2. Compute the loss between the model's output and the true data (e.g., MSE loss for regression tasks)
            loss.backward()                                                 # 3. Backpropagate to compute gradients w.r.t. input images
        
        elif in3 is not None:
            loss = self.recon_loss1(out_x, in1) + self.recon_loss3(out_z, in3)
            loss.backward()

        elif in2 is not None:
            loss = self.recon_loss1(out_x, in1) + self.recon_loss2(out_y, in2)
            loss.backward()

        else:
            loss = self.recon_loss1(out_x, in1)
            loss.backward()

        
        if self.vta[0] and self.vta[1]:
            signed_grad_x = x.grad.sign()                               # 4.a. Take the sign of the gradients to get the direction of maximum increase in loss for view 1
            signed_grad_y = y.grad.sign() if y is not None else None
            x = x + epsilon * signed_grad_x                             #   b. Create adversarial images for view 1 by adding a small perturbation (𝜀) in the direction of the gradient
            y = y + epsilon * signed_grad_y if y is not None else None
            x = torch.clamp(x, 0, 1)                                    #   c. Ensure the adversarial images for view 1 are valid pixel values (between 0 and 1)
            y = torch.clamp(y, 0, 1) if y is not None else None
            return x.detach(), y.detach(), signed_grad_x.detach(), signed_grad_y.detach()

        elif self.vta[0]:
            print(x)
            print(x.grad)
            signed_grad = x.grad.sign()                                 # 4.a.
            x = x + epsilon * signed_grad                               #   b.
            x = torch.clamp(x, 0, 1)                                    #   c.
            return x.detach(), signed_grad.detach()

        elif self.vta[1]:
            signed_grad = y.grad.sign() if y is not None else None      # 4.a.
            y = y + epsilon * signed_grad if y is not None else None    #   b.
            y = torch.clamp(y, 0, 1) if y is not None else None         #   c.
            return y.detach(), signed_grad.detach()

    def create_adversarial_batch_bim(self, in1, in2=None, in3=None, epsilon=0.1, alpha=0.01, num_iterations=10):
        if self.vta[0] and self.vta[1]:
            x = in1.clone().detach().to(self.opt.device)
            y = in2.clone().detach().to(self.opt.device) if in2 is not None else None
            z = in3.to(self.opt.device) if in3 is not None else None

        elif self.vta[0]:
            x = in1.clone().detach().to(self.opt.device)
            y = in2.to(self.opt.device) if in2 is not None else None
            z = in3.to(self.opt.device) if in3 is not None else None

        elif self.vta[1]:
            x = in1.to(self.opt.device)
            y = in2.clone().detach().to(self.opt.device) if in2 is not None else None
            z = in3.to(self.opt.device) if in3 is not None else None


        image_plus = in1 + epsilon
        image_minus = in1 - epsilon
        sketch_plus = in2 + epsilon if in2 is not None else None
        sketch_minus = in2 - epsilon if in2 is not None else None

        for _ in range(num_iterations):
            if self.vta[0] and self.vta[1]:
                x = x.clone().detach()
                y = y.clone().detach() if y is not None else None
                x.requires_grad_(True)
                y.requires_grad_(True) if y is not None else None

            elif self.vta[0]:
                x = x.clone().detach()
                x.requires_grad_(True)

            elif self.vta[1]:
                y = y.clone().detach() if y is not None else None
                y.requires_grad_(True) if y is not None else None

            out_x, out_y, out_z = self.run_model_batch(x, y, z)

            if in3 is not None and in2 is not None:
                recon_loss1 = torch.nn.MSELoss()
                recon_loss2 = torch.nn.MSELoss()
                recon_loss3 = torch.nn.BCEWithLogitsLoss()

                loss = recon_loss1(out_x, in1) + recon_loss2(out_y, in2) + recon_loss3(out_z, in3)
                loss.backward()

            elif in3 is not None:
                recon_loss1 = torch.nn.MSELoss()
                recon_loss3 = torch.nn.BCEWithLogitsLoss()

                loss = recon_loss1(out_x, in1) + recon_loss3(out_z, in3)
                loss.backward()

            elif in2 is not None:
                recon_loss1 = torch.nn.MSELoss()
                recon_loss2 = torch.nn.MSELoss()

                loss = recon_loss1(out_x, in1) + recon_loss2(out_y, in2)
                loss.backward()
            else:
                recon_loss1 = torch.nn.MSELoss()

                loss = recon_loss1(out_x, in1)
                loss.backward()

            if self.vta[0] and self.vta[1]:
                signed_grad_x = x.grad.sign()
                signed_grad_y = y.grad.sign() if y is not None else None

                x = x + alpha * signed_grad_x
                y = y + alpha * signed_grad_y if y is not None else None

                x = torch.max(image_minus, torch.min(image_plus, x))
                y = torch.max(sketch_minus, torch.min(sketch_plus, y))

                x = torch.clamp(x, 0, 1)
                y = torch.clamp(y, 0, 1) if y is not None else None
                return x.detach(), y.detach(), signed_grad_x.detach(), signed_grad_y.detach()

            elif self.vta[0]:
                signed_grad = x.grad.sign()
                x = x + alpha * signed_grad
                x = torch.max(image_minus, torch.min(image_plus, x))
                x = torch.clamp(x, 0, 1)
                return x.detach(), signed_grad.detach()

            elif self.vta[1]:
                signed_grad = y.grad.sign() if y is not None else None
                y = y + alpha * signed_grad if y is not None else None
                y = torch.max(sketch_minus, torch.min(sketch_plus, y)) if y is not None else None
                y = torch.clamp(y, 0, 1) if y is not None else None
                return y.detach(), signed_grad.detach()