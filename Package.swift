// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.
import PackageDescription

let package = Package(
    name: "FacialAuthFramework",
    platforms: [
        .iOS(.v14)
    ],
    products: [
        // Framework principal
        .library(
            name: "FacialAuthFramework",
            targets: ["FacialAuthFramework"]
        ),
    ],
    dependencies: [],
    targets: [
        // Target principal del framework
        .target(
            name: "FacialAuthFramework",
            dependencies: [],
            path: "Sources/FacialAuthFramework",
            resources: [
                // Solo process - incluye todos los archivos de Resources
                .process("Resources")
            ],
            swiftSettings: [
                .define("SWIFT_PACKAGE")
            ]
        ),
        
         //Tests unitarios
        .testTarget(
            name: "FacialAuthFrameworkTests",
            dependencies: ["FacialAuthFramework"],
            path: "Tests/FacialAuthFrameworkTests"
        ),
    ]
)
