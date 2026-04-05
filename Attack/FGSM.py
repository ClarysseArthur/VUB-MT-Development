import torch


class FGSM:
    def __init__(self, net_in1, net_out1, net_in2=None, net_out2=None, net_in3=None, net_out3=None, kPCA=None, opt=None):
        self.net_in1 = net_in1
        self.net_in2 = net_in2
        self.net_in3 = net_in3

        self.net_out1 = net_out1
        self.net_out2 = net_out2
        self.net_out3 = net_out3

        self.kPCA = kPCA
        self.opt = opt

        self.net_in1.zero_grad()
        self.net_in2.zero_grad() if self.net_in2 is not None else None
        self.net_in3.zero_grad() if self.net_in3 is not None else None
        self.net_out1.zero_grad()
        self.net_out2.zero_grad() if self.net_out2 is not None else None
        self.net_out3.zero_grad() if self.net_out3 is not None else None

    def _kPCA(self, x_en, y_en=None, z_en=None):
        if y_en is not None and z_en is not None:
            return self.kPCA(x_en, y_en, z_en)
        elif y_en is not None:
            return self.kPCA(x_en, y_en)
        else:
            return self.kPCA(x_en)

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

    def create_adversarial_batch_fgsm(self, in1, in2=None, in3=None, epsilon=0.1):
        x_adv = in1.clone().detach().to(self.opt.device)
        x_adv.requires_grad_(True)

        y = in2.to(self.opt.device) if in2 is not None else None
        z = in3.to(self.opt.device) if in3 is not None else None

        out_x, _, _ = self.run_model_batch(x_adv, y, z)             # 1. Forward clean image batch through the model

        loss = torch.nn.functional.mse_loss(out_x, in1)             # 2. Compute reconstruction loss between output and original input
        loss.backward()                                             # 3. Backpropagate to compute gradients w.r.t. input images

        signed_grad = x_adv.grad.sign()                             # 4.a. Take the sign of the gradients to get the direction of maximum increase in loss
        x_adv = x_adv + epsilon * signed_grad                       #   b. Create adversarial images by adding a small perturbation (𝜀) in the direction of the gradient
        x_adv = torch.clamp(x_adv, 0, 1)                            #   c. Ensure the adversarial images are valid pixel values (between 0 and 1)

        return x_adv.detach(), signed_grad.detach()

    def create_adversarial_batch_bim(self, in1, in2=None, in3=None, epsilon=0.1, alpha=0.01, num_iterations=10):
        adv_x = in1.clone().detach().to(self.opt.device)
        y = in2.to(self.opt.device) if in2 is not None else None
        z = in3.to(self.opt.device) if in3 is not None else None

        image_plus = in1 + epsilon
        image_minus = in1 - epsilon

        for _ in range(num_iterations):
            adv_x = adv_x.clone().detach()
            adv_x.requires_grad_(True)

            out_x, _, _ = self.run_model_batch(adv_x, y, z)

            loss = torch.nn.functional.mse_loss(out_x, in1)
            loss.backward()

            signed_grad = adv_x.grad.sign()

            # Iterative FGSM step
            image_prime = adv_x + alpha * signed_grad

            # Clip to epsilon-ball around original image, then to valid image range
            adv_x = torch.max(image_minus, image_prime)
            adv_x = torch.min(image_plus, adv_x)
            adv_x = torch.clamp(adv_x, 0, 1)

        return adv_x.detach(), signed_grad.detach()