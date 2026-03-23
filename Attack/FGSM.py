import torch


class FGSM_1V:
    def __init__(self, net_in, net_out, kPCA, opt):
        self.net_in = net_in
        self.net_out = net_out
        self.kPCA = kPCA
        self.opt = opt

    def run_model_batch_1V(self, img_batch):
        x_en = self.net_in(img_batch)

        h, _ = self.kPCA(x_en)
        U = torch.mm(x_en.t(), h)

        x_tilde = self.net_out(torch.mm(h, U.t()))
        return x_tilde

    def create_adversarial_batch(self, in_images, epsilon=0.1):
        self.net_in.zero_grad()
        self.net_out.zero_grad()

        adv_images = in_images.clone().detach().to(self.opt.device)
        adv_images.requires_grad_(True)

        out_images = self.run_model_batch_1V(adv_images)

        loss = torch.nn.functional.mse_loss(out_images, in_images)
        loss.backward()

        signed_grad = adv_images.grad.sign()
        adv_images = adv_images + epsilon * signed_grad
        adv_images = torch.clamp(adv_images, 0, 1)

        return adv_images.detach(), signed_grad.detach()

    def create_adversarial_batch_bim(self, in_images, epsilon=0.1, alpha=0.01, num_iterations=10):
        adv_images = in_images.clone().detach().to(self.opt.device)

        image_plus = in_images + epsilon
        image_minus = in_images - epsilon

        for _ in range(num_iterations):
            self.net_in1.zero_grad()
            self.net_out1.zero_grad()

            adv_images = adv_images.clone().detach()
            adv_images.requires_grad_(True)

            out_images, _ = self.run_model_batch(adv_images)

            loss = torch.nn.functional.mse_loss(out_images, in_images)
            loss.backward()

            signed_grad = adv_images.grad.sign()

            # Iterative FGSM step
            image_prime = adv_images + alpha * signed_grad

            # Clip to epsilon-ball around original image, then to valid image range
            adv_images = torch.max(image_minus, image_prime)
            adv_images = torch.min(image_plus, adv_images)
            adv_images = torch.clamp(adv_images, 0, 1)

        return adv_images.detach(), signed_grad.detach()


class FGSM_2V:
    def __init__(self, net_in1, net_in2, net_out1, net_out2, kPCA, opt):
        self.net_in1 = net_in1
        self.net_in2 = net_in2
        self.net_out1 = net_out1
        self.net_out2 = net_out2
        self.kPCA = kPCA
        self.opt = opt

    def run_model_batch(self, img_batch, label_batch):
        x_en = self.net_in1(img_batch)
        y_en = self.net_in2(label_batch)

        h, _ = self.kPCA(x_en, y_en)
        U = torch.mm(x_en.t(), h)
        V = torch.mm(y_en.t(), h)

        x_tilde = self.net_out1(torch.mm(h, U.t()))
        y_tilde = self.net_out2(torch.mm(h, V.t()))
        return x_tilde, y_tilde

    def create_adversarial_batch(self, in_images, in_labels, epsilon=0.1):
        self.net_in1.zero_grad()
        self.net_in2.zero_grad()
        self.net_out1.zero_grad()
        self.net_out2.zero_grad()

        adv_images = in_images.clone().detach().to(self.opt.device)
        adv_images.requires_grad_(True)
        in_labels = in_labels.to(self.opt.device)

        out_images, _ = self.run_model_batch(adv_images, in_labels) # 1. Forward clean image batch through the model

        loss = torch.nn.functional.mse_loss(out_images, in_images)  # 2. Compute reconstruction loss between output and original input
        loss.backward()                                             # 3. Backpropagate to compute gradients w.r.t. input images

        signed_grad = adv_images.grad.sign()                        # 4.a. Take the sign of the gradients to get the direction of maximum increase in loss
        adv_images = adv_images + epsilon * signed_grad             # 4.b. Create adversarial images by adding a small perturbation (𝜀) in the direction of the gradient
        adv_images = torch.clamp(adv_images, 0, 1)                  # 4.c. Ensure the adversarial images are valid pixel values (between 0 and 1)

        return adv_images.detach(), signed_grad.detach()

    def create_adversarial_batch_bim(self, in_images, in_labels, epsilon=0.1, alpha=0.01, num_iterations=10):
        adv_images = in_images.clone().detach().to(self.opt.device)
        in_labels = in_labels.clone().detach().to(self.opt.device)

        image_plus = in_images + epsilon
        image_minus = in_images - epsilon

        for _ in range(num_iterations):
            self.net_in1.zero_grad()
            self.net_in2.zero_grad()
            self.net_out1.zero_grad()
            self.net_out2.zero_grad()

            adv_images = adv_images.clone().detach()
            adv_images.requires_grad_(True)

            out_images, _ = self.run_model_batch(adv_images, in_labels)

            loss = torch.nn.functional.mse_loss(out_images, in_images)
            loss.backward()

            signed_grad = adv_images.grad.sign()

            # Iterative FGSM step
            image_prime = adv_images + alpha * signed_grad

            # Clip to epsilon-ball around original image, then to valid image range
            adv_images = torch.max(image_minus, image_prime)
            adv_images = torch.min(image_plus, adv_images)
            adv_images = torch.clamp(adv_images, 0, 1)

        return adv_images.detach(), signed_grad.detach()