import os
import json
import random
import SimpleITK as sitk
import cupy as cp

# Function to register CT images using CUDA
def register_ct_images_gpu(fixed_image_path, moving_image_path, output_path):
    """
    Registers two CT-scan images using SimpleITK with CUDA acceleration.

    Args:
        fixed_image_path: Path to the fixed image (reference image).
        moving_image_path: Path to the moving image.
        output_path: Path to save the registered moving image.

    Returns:
        None
    """
    fixed_image = sitk.ReadImage(fixed_image_path, sitk.sitkFloat32)
    moving_image = sitk.ReadImage(moving_image_path, sitk.sitkFloat32)

    # Convert images to cupy arrays
    fixed_image_data = cp.asarray(sitk.GetArrayFromImage(fixed_image))
    moving_image_data = cp.asarray(sitk.GetArrayFromImage(moving_image))

    # Create SimpleITK images from cupy arrays
    fixed_image_gpu = sitk.GetImageFromArray(fixed_image_data.get())
    fixed_image_gpu.CopyInformation(fixed_image)
    moving_image_gpu = sitk.GetImageFromArray(moving_image_data.get())
    moving_image_gpu.CopyInformation(moving_image)

    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=0.1, minStep=1e-4, numberOfIterations=100, gradientMagnitudeTolerance=1e-3
    )
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetInterpolator(sitk.sitkLinear)

    initial_transform = sitk.CenteredTransformInitializer(
        fixed_image_gpu,
        moving_image_gpu,
        sitk.AffineTransform(3),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )

    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    final_transform = registration_method.Execute(fixed_image_gpu, moving_image_gpu)

    resampled_image_gpu = sitk.Resample(
        moving_image_gpu,
        fixed_image_gpu,
        final_transform,
        sitk.sitkLinear,
        0.0,
        moving_image_gpu.GetPixelID()
    )

    # Convert cupy array back to numpy array
    resampled_image_data = cp.asnumpy(sitk.GetArrayFromImage(resampled_image_gpu))

    # Create SimpleITK image from numpy array
    resampled_image = sitk.GetImageFromArray(resampled_image_data)
    resampled_image.CopyInformation(fixed_image)

    sitk.WriteImage(resampled_image, output_path)

# Load the JSON file
if __name__ == '__main__':
    with open('dataset.json', 'r') as f:
        data = json.load(f)

    # Extract training images
    training_images = [entry['image'] for entry in data['training']]

    # Output directory
    output_dir = '../registered_images_gpu'
    os.makedirs(output_dir, exist_ok=True)

    # Perform registration for each moving image
    for moving_image in training_images:
        # Create a subfolder for the moving image
        moving_image_name = os.path.basename(moving_image).split('.')[0]
        moving_image_folder = os.path.join(output_dir, moving_image_name)
        os.makedirs(moving_image_folder, exist_ok=True)

        # Randomly select 10 fixed images
        fixed_images = random.sample(training_images, 10)
        print("Random Path:", fixed_images)

        # Actual Path
        moving_image = os.path.join('../MICCAI/', moving_image[2:])

        # Save the current CT-Scan
        current_image = sitk.ReadImage(moving_image, sitk.sitkFloat32)
        sitk.WriteImage(current_image, os.path.join(moving_image_folder, f'{1}.nii.gz'))

        # Register the moving image with each fixed image
        j=1
        for _, fixed_image in enumerate(fixed_images):
            fixed_image = os.path.join('../MICCAI/', fixed_image[2:])
            if(fixed_image!=moving_image):
                j+=1
                output_path = os.path.join(moving_image_folder, f'{j}.nii.gz')
                register_ct_images_gpu(fixed_image, moving_image, output_path)
                if j==10:
                    break

    print("Registration completed and images saved.")