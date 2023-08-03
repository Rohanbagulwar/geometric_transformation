import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import time


class ImageTransformer:
    """A class used to apply geometric transformations to images."""

    def __init__(self, image):
        """Initializes ImageTransformer with an image.

        Args:
            image (np.array): The image to transform.
        """
        self.image = image

    def scale(self, fx, fy):
        """Scales the image.

        Args:
            fx (float): Scale factor along the horizontal axis.
            fy (float): Scale factor along the vertical axis.

        Returns:
            np.array: The scaled image.
        """
        return cv2.resize(self.image, None, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)

    def rotate(self, angle):
        """Rotates the image.

        Args:
            angle (float): The angle of rotation.

        Returns:
            np.array: The rotated image.
        """
        rows, cols = self.image.shape[:2]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        return cv2.warpAffine(self.image, M, (cols, rows))

    def affine_transform(self, pts1, pts2):
        """Applies an affine transformation to the image.

        Args:
            pts1 (np.array): Input points for affine transformation.
            pts2 (np.array): Output points for affine transformation.

        Returns:
            np.array: The transformed image.
        """
        M = cv2.getAffineTransform(pts1, pts2)
        return cv2.warpAffine(self.image, M, (self.image.shape[1], self.image.shape[0]))

    def translate(self, dx, dy):
        """Translates the image.

        Args:
            dx (int): Translation along the x axis.
            dy (int): Translation along the y axis.

        Returns:
            np.array: The translated image.
        """
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        return cv2.warpAffine(self.image, M, (self.image.shape[1], self.image.shape[0]))

    def projective(self, pts1, pts2):
        """Applies a projective transformation to the image.

        Args:
            pts1 (np.array): Input points for projective transformation.
            pts2 (np.array): Output points for projective transformation.

        Returns:
            np.array: The transformed image.
        """
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        return cv2.warpPerspective(self.image, matrix, (self.image.shape[1], self.image.shape[0]))


def main():
    st.title('OpenCV Geometric Transformations')

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        transformer = ImageTransformer(img_array)

        option = st.selectbox(
            'Which geometric transformation would you like to apply?',
            ('Scaling', 'Rotation', 'Affine Transformation', 'Translation', 'Projective'))

        col1, col2 = st.columns(2)
        col1.header("Original Image")
        col1.image(image, use_column_width=True)

        if option == 'Scaling':
            fx = st.slider('Scale factor in x direction', 0.1, 3.0, 1.0)
            fy = st.slider('Scale factor in y direction', 0.1, 3.0, 1.0)
            result = transformer.scale(fx, fy)
            output_name = 'output_scaled.png'
        elif option == 'Rotation':
            angle = st.slider('Angle of rotation', 0, 360, 180)
            result = transformer.rotate(angle)
            output_name = 'output_rotated.png'
        elif option == 'Affine Transformation':
            st.markdown(
                'Affine transformation requires 3 points in the source (input) image and their corresponding locations in the output image. The transformation will move the input points to the output points, changing the image perspective.')
            st.markdown(
                'The input points are set to the corners and center of the image. By changing the output points, you can stretch, shrink, rotate, and shear the image.')

            # Input points are set to the corners and center of the image
            pts1 = np.float32([[0, 0], [image.width, 0], [image.width / 2, image.height / 2]])

            # User can modify output points to transform the image
            st.subheader('Output Points')
            pt1 = (st.number_input('Output x1', value=0), st.number_input('Output y1', value=0))
            pt2 = (st.number_input('Output x2', value=image.width), st.number_input('Output y2', value=0))
            pt3 = (st.number_input('Output x3', value=image.width / 2), st.number_input('Output y3', value=image.height / 2))
            pts2 = np.float32([pt1, pt2, pt3])

            result = transformer.affine_transform(pts1, pts2)
            output_name = 'output_affine_transformed.png'

        elif option == 'Translation':
            dx = st.slider('Translation in x direction', -500, 500, 0)
            dy = st.slider('Translation in y direction', -500, 500, 0)
            result = transformer.translate(dx, dy)
            output_name = 'output_translated.png'
        else:  # Projective
            st.markdown(
                'Projective transformation requires 4 points in the source (input) image and their corresponding locations in the output image.')
            st.markdown(
                'By default, the points are set to the corners of the image. By changing the output points, you can distort the image.')

            # Input points are set to the corners of the image
            pts1 = np.float32([[0, 0], [0, image.height], [image.width, 0], [image.width, image.height]])

            # User can modify output points to distort the image
            st.subheader('Output Points')
            pt1 = (st.number_input('Output x1', value=0), st.number_input('Output y1', value=0))
            pt2 = (st.number_input('Output x2', value=0), st.number_input('Output y2', value=image.height))
            pt3 = (st.number_input('Output x3', value=image.width), st.number_input('Output y3', value=0))
            pt4 = (st.number_input('Output x4', value=image.width), st.number_input('Output y4', value=image.height))
            pts2 = np.float32([pt1, pt2, pt3, pt4])

            result = transformer.projective(pts1, pts2)
            output_name = 'output_projective.png'

        col2.header("Transformed Image")
        col2.image(result, use_column_width=True)

        if st.button('Save Output Image'):
            # Get the current timestamp to avoid overwriting previous files
            timestamp = time.strftime("%Y%m%d-%H%M%S")

            # Use the transformation type, original image name, and timestamp to create a unique output name
            output_name = f'{option}_{uploaded_file.name.split(".")[0]}_{timestamp}.png'

            # Convert the image to a format that can be downloaded
            result_pil = Image.fromarray(result)
            with io.BytesIO() as output:
                result_pil.save(output, format="PNG")
                binary_data = output.getvalue()

            # Create a download button for the image
            st.download_button(
                label="Download Output Image",
                data=binary_data,
                file_name=output_name,
                mime="image/png",
            )


if __name__ == "__main__":
    main()
