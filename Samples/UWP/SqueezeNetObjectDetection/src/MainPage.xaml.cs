using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Windows.UI.Xaml;
using Windows.UI.Xaml.Controls;
using Windows.AI.MachineLearning.Preview;
using Windows.Storage;
using Windows.Media;
using Windows.Graphics.Imaging;
using System.Threading.Tasks;
using Windows.Storage.Streams;
using Windows.UI.Core;
using Windows.Storage.Pickers;
using Windows.UI.Xaml.Media.Imaging;
using Windows.UI.Xaml.Navigation;
using Windows.Media.Capture;
using Windows.Media.MediaProperties;
using Windows.System.Threading;
using System.Threading;
using Windows.Media.Devices;
using Windows.Devices.Enumeration;
using System.Diagnostics;
using Windows.Media.SpeechSynthesis;

namespace SqueezeNetObjectDetection
{
    public sealed partial class MainPage : Page
    {
        private const string _kModelFileName = "SqueezeNet.onnx";
        private const string _kLabelsFileName = "Labels.json";
        private ImageVariableDescriptorPreview _inputImageDescription;
        private TensorVariableDescriptorPreview _outputTensorDescription;
        private LearningModelPreview _model = null;
        private List<string> _labels = new List<string>();
        private List<float> _outputVariableList = new List<float>();

        private MediaCapture _captureManager;
        private VideoEncodingProperties _videoProperties;
        private ThreadPoolTimer _frameProcessingTimer;
        private SemaphoreSlim _frameProcessingSemaphore = new SemaphoreSlim(1);
        private SpeechSynthesizer _speechSynth;

        public MainPage()
        {
            InitializeComponent();
        }

        protected override async void OnNavigatedTo(NavigationEventArgs e)
        {
            base.OnNavigatedTo(e);

            await LoadModelAsync();
        }

        /// <summary>
        /// Load the label and model files
        /// </summary>
        /// <returns></returns>
        private async Task LoadModelAsync(bool isGpu = false)
        {
            await Dispatcher.RunAsync(CoreDispatcherPriority.Normal, () => StatusBlock.Text = $"Loading {_kModelFileName} ... patience ");

            try
            {
                // Parse labels from label file
                var file = await StorageFile.GetFileFromApplicationUriAsync(new Uri($"ms-appx:///Assets/{_kLabelsFileName}"));
                using (var inputStream = await file.OpenReadAsync())
                using (var classicStream = inputStream.AsStreamForRead())
                using (var streamReader = new StreamReader(classicStream))
                {
                    string line = "";
                    char[] charToTrim = { '\"', ' ' };
                    while (streamReader.Peek() >= 0)
                    {
                        line = streamReader.ReadLine();
                        line.Trim(charToTrim);
                        var indexAndLabel = line.Split(':');
                        if (indexAndLabel.Count() == 2)
                        {
                            _labels.Add(indexAndLabel[1]);
                        }
                    }
                }

                // Load Model
                var modelFile = await StorageFile.GetFileFromApplicationUriAsync(new Uri($"ms-appx:///Assets/{_kModelFileName}"));
                _model = await LearningModelPreview.LoadModelFromStorageFileAsync(modelFile);
                _model.InferencingOptions.ReclaimMemoryAfterEvaluation = true;
                _model.InferencingOptions.PreferredDeviceKind = isGpu == true ? LearningModelDeviceKindPreview.LearningDeviceGpu : LearningModelDeviceKindPreview.LearningDeviceCpu;

                // Retrieve model input and output variable descriptions (we already know the model takes an image in and outputs a tensor)
                List<ILearningModelVariableDescriptorPreview> inputFeatures = _model.Description.InputFeatures.ToList();
                List<ILearningModelVariableDescriptorPreview> outputFeatures = _model.Description.OutputFeatures.ToList();

                _inputImageDescription =
                    inputFeatures.FirstOrDefault(feature => feature.ModelFeatureKind == LearningModelFeatureKindPreview.Image)
                    as ImageVariableDescriptorPreview;

                _outputTensorDescription =
                    outputFeatures.FirstOrDefault(feature => feature.ModelFeatureKind == LearningModelFeatureKindPreview.Tensor)
                    as TensorVariableDescriptorPreview;

                await Dispatcher.RunAsync(CoreDispatcherPriority.Normal, () => StatusBlock.Text = $"Loaded {_kModelFileName}. Go ahead and press the camera button.");

            }
            catch (Exception ex)
            {
                await Dispatcher.RunAsync(CoreDispatcherPriority.Normal, () => StatusBlock.Text = $"error: {ex.Message}");
                _model = null;
            }
        }

        private async void OnWebCameraButtonClicked(object sender, RoutedEventArgs e)
        {
            if (_captureManager == null || _captureManager.CameraStreamState != CameraStreamState.Streaming)
            {
                await StartWebCameraAsync();
            }
            else
            {
                await StopWebCameraAsync();
            }
        }

        private async void OnDeviceToggleToggled(object sender, RoutedEventArgs e)
        {
            await LoadModelAsync(DeviceToggle.IsOn);
        }


        /// <summary>
        /// Event handler for camera source changes
        /// </summary>
        private async Task StartWebCameraAsync()
        {
            try
            {
                if (_captureManager == null ||
                    _captureManager.CameraStreamState == CameraStreamState.Shutdown ||
                    _captureManager.CameraStreamState == CameraStreamState.NotStreaming)
                {
                    if (_captureManager != null)
                    {
                        _captureManager.Dispose();
                    }

                    // Workaround since my home built-in camera does not work as expected, so have to use my LifeCam
                    MediaCaptureInitializationSettings settings = new MediaCaptureInitializationSettings();
                    var allCameras = await DeviceInformation.FindAllAsync(DeviceClass.VideoCapture);
                    var selectedCamera = allCameras.FirstOrDefault(c => c.Name.Contains("LifeCam")) ?? allCameras.FirstOrDefault();
                    if (selectedCamera != null)
                    {
                        settings.VideoDeviceId = selectedCamera.Id;
                    }
                    //settings.PreviewMediaDescription = new MediaCaptureVideoProfileMediaDescription()

                    _captureManager = new MediaCapture();
                    await _captureManager.InitializeAsync(settings);
                    WebCamCaptureElement.Source = _captureManager;
                }

                if (_captureManager.CameraStreamState == CameraStreamState.NotStreaming)
                {
                    if (_frameProcessingTimer != null)
                    {
                        _frameProcessingTimer.Cancel();
                        _frameProcessingSemaphore.Release();
                    }

                    TimeSpan timerInterval = TimeSpan.FromMilliseconds(66); //15fps
                    _frameProcessingTimer = ThreadPoolTimer.CreatePeriodicTimer(new TimerElapsedHandler(ProcessCurrentVideoFrame), timerInterval);

                    _videoProperties = _captureManager.VideoDeviceController.GetMediaStreamProperties(MediaStreamType.VideoPreview) as VideoEncodingProperties;

                    await _captureManager.StartPreviewAsync();

                    WebCamCaptureElement.Visibility = Visibility.Visible;
                }
            }
            catch (Exception ex)
            {
                await Dispatcher.RunAsync(CoreDispatcherPriority.Normal, () => StatusBlock.Text = $"error: {ex.Message}");
            }
        }

        public async Task StopWebCameraAsync()
        {
            try
            {
                if (_frameProcessingTimer != null)
                {
                    _frameProcessingTimer.Cancel();
                }

                if (_captureManager != null && _captureManager.CameraStreamState != CameraStreamState.Shutdown)
                {
                    await _captureManager.StopPreviewAsync();
                    WebCamCaptureElement.Source = null;
                    _captureManager.Dispose();
                    _captureManager = null;

                    WebCamCaptureElement.Visibility = Visibility.Collapsed;
                }
            }
            catch (Exception ex)
            {
                await Dispatcher.RunAsync(CoreDispatcherPriority.Normal, () => StatusBlock.Text = $"error: {ex.Message}");
            }
        }

        private async void ProcessCurrentVideoFrame(ThreadPoolTimer timer)
        {
            if (_captureManager.CameraStreamState != CameraStreamState.Streaming || !_frameProcessingSemaphore.Wait(0))
            {
                return;
            }

            try
            {
                const BitmapPixelFormat InputPixelFormat = BitmapPixelFormat.Bgra8;
                using (VideoFrame previewFrame = new VideoFrame(InputPixelFormat, (int)_videoProperties.Width, (int)_videoProperties.Height))
                {
                    await _captureManager.GetPreviewFrameAsync(previewFrame);
                    await EvaluateVideoFrameAsync(previewFrame);
                }
            }
            catch (Exception ex)
            {
                await Dispatcher.RunAsync(CoreDispatcherPriority.Normal, () => StatusBlock.Text = $"error: {ex.Message}");
            }
            finally
            {
                _frameProcessingSemaphore.Release();
            }
        }

        //private async Task EvaluateImageAsync(Func<Task<StorageFile>> loadImageAction)
        //{
        //    try
        //    {
        //        // Load the model
        //        await Task.Run(async () => await LoadModelAsync());

        //        // Load image
        //        SoftwareBitmap softwareBitmap;
        //        var selectedStorageFile = await loadImageAction();
        //        using (IRandomAccessStream stream = await selectedStorageFile.OpenAsync(FileAccessMode.Read))
        //        {
        //            // Create the decoder from the stream 
        //            BitmapDecoder decoder = await BitmapDecoder.CreateAsync(stream);

        //            // Get the SoftwareBitmap representation of the file in BGRA8 format
        //            softwareBitmap = await decoder.GetSoftwareBitmapAsync();
        //            softwareBitmap = SoftwareBitmap.Convert(softwareBitmap, BitmapPixelFormat.Bgra8, BitmapAlphaMode.Premultiplied);
        //        }

        //        // Display the image
        //        SoftwareBitmapSource imageSource = new SoftwareBitmapSource();
        //        await imageSource.SetBitmapAsync(softwareBitmap);

        //        // Encapsulate the image within a VideoFrame to be bound and evaluated
        //        VideoFrame inputImage = VideoFrame.CreateWithSoftwareBitmap(softwareBitmap);

        //        await Task.Run(async () =>
        //        {
        //            // Evaluate the image
        //            await EvaluateVideoFrameAsync(inputImage);
        //        });
        //    }
        //    catch (Exception ex)
        //    {
        //        await Dispatcher.RunAsync(CoreDispatcherPriority.Normal, () => StatusBlock.Text = $"error: {ex.Message}");
        //    }
        //}


        /// <summary>
        /// Evaluate the VideoFrame passed in as arg
        /// </summary>
        /// <param name="inputFrame"></param>
        /// <returns></returns>
        private async Task EvaluateVideoFrameAsync(VideoFrame inputFrame)
        {
            if (inputFrame != null)
            {
                try
                {
                    // Create bindings for the input and output buffer
                    LearningModelBindingPreview binding = new LearningModelBindingPreview(_model as LearningModelPreview);
                    binding.Bind(_inputImageDescription.Name, inputFrame);
                    binding.Bind(_outputTensorDescription.Name, _outputVariableList);

                    // Process the frame with the model
                    var stopwatch = Stopwatch.StartNew();
                    LearningModelEvaluationResultPreview results = await _model.EvaluateAsync(binding, "test");
                    stopwatch.Stop();
                    List<float> resultProbabilities = results.Outputs[_outputTensorDescription.Name] as List<float>;

                    // Find the result of the evaluation in the bound output (the top classes detected with the max confidence)
                    const int TopPropsCnt = 5;
                    var topProbabilities = new float[TopPropsCnt];
                    var topProbabilityLabelIndexes = new int[TopPropsCnt];
                    for (int i = 0; i < resultProbabilities.Count(); i++)
                    {
                        for (int j = 0; j < TopPropsCnt; j++)
                        {
                            if (resultProbabilities[i] > topProbabilities[j])
                            {
                                topProbabilityLabelIndexes[j] = i;
                                topProbabilities[j] = resultProbabilities[i];
                                break;
                            }
                        }
                    }

                    // Display the result
                    string message = "Predominant detected objects:";
                    for (int i = 0; i < TopPropsCnt; i++)
                    {
                        message += $"\n{topProbabilities[i]*100,5:f0}% : { _labels[topProbabilityLabelIndexes[i]]} ";
                    }
                    await Dispatcher.RunAsync(CoreDispatcherPriority.Normal, async () =>
                    {
                        Duration.Text = $"{1000 / stopwatch.ElapsedMilliseconds,4:f1} fps";
                        StatusBlock.Text = message;
                        await SpeechOutput(_labels[topProbabilityLabelIndexes[0]], topProbabilities[0]);
                    });
                }
                catch (Exception ex)
                {
                    await Dispatcher.RunAsync(CoreDispatcherPriority.Normal, () => StatusBlock.Text = $"error: {ex.Message}");
                }

            }
        }

        private async Task SpeechOutput(string label, float probability)
        {
            if (!SpeechToggle.IsOn || SpeechMediaElement.CurrentState == Windows.UI.Xaml.Media.MediaElementState.Playing)
            {
                return;
            }

            var text = string.Format("This {0} a {1}", probability > 0.75f ? "is likely" : "might be", label);

            // The object for controlling the speech synthesis engine (voice).
            if (_speechSynth == null)
            {
                _speechSynth = new SpeechSynthesizer();
            }

            // Generate the audio stream from plain text.
            SpeechSynthesisStream stream = await _speechSynth.SynthesizeTextToStreamAsync(text);

            // Send the stream to the media object.
            SpeechMediaElement.SetSource(stream, stream.ContentType);
            SpeechMediaElement.Play();
        }

    }
}