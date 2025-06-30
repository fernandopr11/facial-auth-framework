// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "FacialAuthFramework",
    platforms: [
        .iOS(.v14),
        .macOS(.v11)
    ],
    products: [
        .library(
            name: "FacialAuthFramework",
            targets: ["FacialAuthFramework"]
        ),
    ],
    dependencies: [],
    targets: [
        .target(
            name: "FacialAuthFramework",
            dependencies: [],
            path: "Sources/FacialAuthFramework",
            resources: [
                .process("Resources")
            ]
        ),
        .testTarget(
            name: "FacialAuthFrameworkTests",
            dependencies: ["FacialAuthFramework"],
            path: "Tests/FacialAuthFrameworkTests"
        ),
    ]
)
