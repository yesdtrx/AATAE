def ssim2(img1, img2, size_average=True):
    mu1 = img1
    mu2 = img2

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2  # element*element

    sigma1_sq = img1 * img1 - mu1_sq
    sigma2_sq = img2 * img2 - mu2_sq
    sigma12 = img1 * img2 - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ssim3(img1, img2, size_average=True):
    ##
    split = 8
    (_, _, img_size, _) = img1.size()
    window_size = img_size//split
    for i in range(split):
        for j in range(split):
            mu1 = img1[:, :, i * window_size:(i + 1) * window_size,j * window_size:(j + 1) * window_size]
            mu2 = img1[:, :, i * window_size:(i + 1) * window_size, j * window_size:(j + 1) * window_size]
            # mu1 = img1
            # mu2 = img2

            mu1_sq = mu1.pow(2)
            mu2_sq = mu2.pow(2)
            mu1_mu2 = mu1 * mu2  # element*element

            sigma1_sq = img1 * img1 - mu1_sq
            sigma2_sq = img2 * img2 - mu2_sq
            sigma12 = img1 * img2 - mu1_mu2

            C1 = 0.01 ** 2
            C2 = 0.03 ** 2

            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
