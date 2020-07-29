//
//  ViewController.swift
//  face_detection
//
//  Created by Shubham Patel on 28/07/20.
//  Copyright © 2020 Shubham Patel. All rights reserved.
//

import UIKit
import AVFoundation
import Vision

class ViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {
    
    private let captureSession = AVCaptureSession()
    private lazy var previewLayer = AVCaptureVideoPreviewLayer(session: self.captureSession)
    private let videoDataOutput = AVCaptureVideoDataOutput()
    private var drawings: [CAShapeLayer] = []
    private let label: UILabel = {
        let label = UILabel()
        label.textColor = .black
        label.translatesAutoresizingMaskIntoConstraints = false
        label.text = "No face detected"
        label.font = label.font.withSize(30)
        return label
    }()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        self.addCameraInput()
        self.showCameraFeed()
        self.getCameraFrames()
        self.captureSession.startRunning()
        setupLabel()
        
    }
    
    private func addCameraInput() {
        guard let device = AVCaptureDevice.DiscoverySession(
            deviceTypes: [.builtInWideAngleCamera],
            mediaType: .video,
            position: .front).devices.first else {
                fatalError("No camera device found")
        }
        let cameraInput = try! AVCaptureDeviceInput(device: device)
        self.captureSession.addInput(cameraInput)
    }
    
    private func showCameraFeed() {
        self.previewLayer.videoGravity = .resizeAspectFill
        self.view.layer.addSublayer(self.previewLayer)
        self.previewLayer.frame = self.view.frame
    }
    
    private func getCameraFrames() {
        self.videoDataOutput.videoSettings = [(kCVPixelBufferPixelFormatTypeKey as NSString): NSNumber(value: kCVPixelFormatType_32BGRA)] as [String: Any]
        self.videoDataOutput.alwaysDiscardsLateVideoFrames = true
        self.videoDataOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "camera_frame_processing_queue"))
        self.captureSession.addOutput(self.videoDataOutput)
        
        guard let connection = self.videoDataOutput.connection(with: AVMediaType.video),
            connection.isVideoOrientationSupported else { return }
        connection.videoOrientation = .portrait
    }
    
    
    
    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard let frame = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            debugPrint("unable to get image from sample buffer")
            return
        }
        
        //face detection
        let faceDetectionRequest = VNDetectFaceLandmarksRequest(completionHandler: { (request: VNRequest, error: Error?) in
            DispatchQueue.main.async {
                if let results = request.results as? [VNFaceObservation], results.count > 0 {
                    
                    
                    //mask detection
                    
                    // load CoreML model
                    guard let model = try? VNCoreMLModel(for: mask_detection().model) else { return }
                    
                    // run an inference with CoreML
                    let request = VNCoreMLRequest(model: model) { (finishedRequest, error) in
                        
                        // grab the inference results
                        guard let results_m = finishedRequest.results as? [VNClassificationObservation] else { return }
                        
                        // grab the highest confidence result
                        guard let Observation = results_m.first else { return }
                        
                        // create the label text components
                        let predclass = "\(Observation.identifier)"
                        
                        // set the label text
                        DispatchQueue.main.async(execute: {
                            if predclass == "Mask off!!" {
                                self.label.textColor = .red
                            }
                            else {
                                self.label.textColor = .black
                            }
                            self.label.text = "\(predclass)"
                        })
                    }
                    
                    guard let pixelBuffer: CVPixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
                    
                    // execute the request
                    try? VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:]).perform([request])
                    
                    
                } else {
                    //self.clearDrawings()
                    self.label.textColor = .black
                    self.label.text = "No face detected"
                }
            }
        })
        let imageRequestHandler = VNImageRequestHandler(cvPixelBuffer: frame, orientation: .leftMirrored, options: [:])
        try? imageRequestHandler.perform([faceDetectionRequest])
    }
    
    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        self.previewLayer.frame = self.view.frame
    }
    
    func setupLabel() {
        self.view.addSubview(self.label)
        self.label.centerXAnchor.constraint(equalTo: self.view.centerXAnchor).isActive = true
        self.label.bottomAnchor.constraint(equalTo: self.view.bottomAnchor, constant: -50).isActive = true
    }
    
}

