# Cerebromark - AI Powered Attendance Intelligence

Cerebromark is an AI-powered attendance management system that leverages computer vision to automate and streamline the process of tracking student presence. By integrating with Hikvision cameras, it provides real-time, accurate attendance data, eliminating the need for manual roll calls and reducing administrative overhead.

## About The Project

This project is an AI-powered system for smart, real-time attendance tracking using advanced computer vision and automation.

### Built With

*   [Python](https://www.python.org/)
*   [Flask](https://flask.palletsprojects.com/)
*   [OpenCV](https://opencv.org/)

## Getting Started

To get a local copy up and running follow these simple example steps.

### Prerequisites

Before you begin, ensure you have the following installed:
*   Python 3.8+
*   pip

### Installation

1.  **Clone the repo**
    ```sh
    git clone https://github.com/Ronin-117/Cerebromark_-_AI_Powered_Attendance_Intelligence.git
    ```
2.  **Navigate to the project directory**
    ```sh
    cd Cerebromark_-_AI_Powered_Attendance_Intelligence
    ```
3.  **Install dependencies**
    ```sh
    pip install -r requirements.txt
    ```
    *Note: If you encounter issues with `dlib`, you may need to install `CMake` and `Visual Studio Build Tools` first.*

## Usage

1.  **Prepare the Student Dataset**:
    *   Create a folder named `NDB`.
    *   Inside `NDB`, create subfolders for each student with the naming convention `registerno_name`.
    *   Place 8-10 images of each student in their respective folders, capturing various angles from a decent distance (close-ups are not necessary).

2.  **Generate Face Embeddings**:
    *   Run the `db_make.py` script to process the images and create the `New_face_db` file, which contains the face embeddings.
    ```sh
    python db_make.py
    ```

3.  **Configure the Application**:
    *   Open `app.py` and update the following:
        *   **Timetable**: Modify the `timetable` dictionary with your class schedule.
        *   **Camera Details**: Enter the IP address, username, and password for your Hikvision camera.

4.  **Run the Application**:
    *   Execute the `app.py` script to start the server.
    ```sh
    python app.py
    ```

5.  **Access the UI**:
    *   Open your web browser and navigate to `http://localhost:5000` to view the application.

## Roadmap

- [ ] Multi-Camera Support
- [ ] Automated Timetable Import
- [ ] Enhanced Reporting
- [ ] Mobile Application
- [ ] Cloud Deployment

See the [open issues](https://github.com/Ronin-117/Cerebromark_-_AI_Powered_Attendance_Intelligence/issues) for a full list of proposed features (and known issues).

## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

Project Link: [https://github.com/Ronin-117/Cerebromark_-_AI_Powered_Attendance_Intelligence](https://github.com/Ronin-117/Cerebromark_-_AI_Powered_Attendance_Intelligence)

## Acknowledgments

*   [OpenCV](https://opencv.org/)
*   [Flask](https://flask.palletsprojects.com/)
*   [Img Shields](https://shields.io)
*   [GitHub Pages](https://pages.github.com)
