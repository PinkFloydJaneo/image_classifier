This project is a web application built using Flask that allows users to upload images for training a face classification model. The application utilizes TensorFlow and Keras for model training and MTCNN for face detection. Users can upload images, train the model, and classify new images based on the trained model.

<!DOCTYPE html>
<html lang="en">
<head>
</head>
<body>

<h1>Instructions for Using the Flask Image Classification App</h1>

<p>This guide will help you set up and use the Flask Image Classification application available in this repository.</p>

<h2>Prerequisites</h2>
<ul>
    <li>Python 3.x installed on your machine</li>
    <li>Basic knowledge of Python and Flask</li>
    <li>Required libraries (listed below)</li>
</ul>

<h2>Required Libraries</h2>
<p>Before running the application, ensure you have the following libraries installed:</p>
<pre><code>pip install Flask tensorflow keras scikit-learn mtcnn Pillow matplotlib numpy</code></pre>

<h2>Setting Up the Project</h2>

<h3>1. Clone the Repository</h3>
<p>Open your terminal and clone the repository using the following command:</p>
<pre><code>git clone https://github.com/yourusername/repository-name.git</code></pre>
<p>Replace <code>yourusername</code> and <code>repository-name</code> with your GitHub username and the name of the repository.</p>

<h3>2. Navigate to the Project Directory</h3>
<p>Change your directory to the cloned repository:</p>
<pre><code>cd repository-name</code></pre>

<h3>3. Create Necessary Directories</h3>
<p>The application requires specific directories for storing images and models. Run the following commands to create them:</p>
<pre><code>mkdir image_classification_dataset</code></pre>
<pre><code>mkdir uploads</code></pre>

<h2>Running the Application</h2>

<h3>1. Start the Flask Server</h3>
<p>Run the application using the following command:</p>
<pre><code>python MultipleFiles/app.py</code></pre>
<p>The server will start, and you should see output indicating that it is running.</p>

<h3>2. Access the Application</h3>
<p>Open your web browser and go to:</p>
<pre><code>http://127.0.0.1:5000/</code></pre>
<p>This will take you to the home page of the application.</p>

<h2>Using the Application</h2>
<ul>
    <li><strong>Upload and Train Model</strong>
        <ul>
            <li>Navigate to the upload page to upload images for training.</li>
            <li>Provide a class name and select images to upload.</li>
            <li>The model will be trained immediately after the upload.</li>
        </ul>
    </li>
    <li><strong>Classify Images</strong>
        <ul>
            <li>Go to the classify page to upload an image for classification.</li>
            <li>The application will extract the face from the image and predict the class.</li>
        </ul>
    </li>
    <li><strong>View Trained Models</strong>
        <ul>
            <li>You can view the list of trained models and delete any model if needed.</li>
        </ul>
    </li>
</ul>

<h2>Notes</h2>
<ul>
    <li>Ensure that the images you upload for training contain faces for better accuracy.</li>
    <li>The application uses MTCNN for face detection, so the images should be clear and well-lit.</li>
</ul>

<h2>Troubleshooting</h2>
<ul>
    <li>If you encounter any issues, check the console for error messages.</li>
    <li>Ensure all required libraries are installed and up to date.</li>
</ul>

<h2>License</h2>
<p>This project is licensed under the MIT License - see the <a href="LICENSE">LICENSE</a> file for details.</p>

</body>
</html>
