plugins {
    id("java")
}

group = "org.example"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

dependencies {
    testImplementation("org.junit.jupiter:junit-jupiter-api:5.8.1")
    testRuntimeOnly("org.junit.jupiter:junit-jupiter-engine:5.8.1")
    implementation("ai.djl:api:0.26.0")
    implementation("ai.djl:model-zoo:0.26.0")
    implementation("ai.djl:basicdataset:0.26.0")
    // https://mvnrepository.com/artifact/commons-cli/commons-cli
    implementation("commons-cli:commons-cli:1.6.0")
// https://mvnrepository.com/artifact/ai.djl.mxnet/mxnet-engine
    runtimeOnly("ai.djl.mxnet:mxnet-engine:0.26.0")
// https://mvnrepository.com/artifact/ai.djl.tensorflow/tensorflow-native-auto
//    runtimeOnly("ai.djl.tensorflow:tensorflow-native-auto:2.4.1")
//     https://mvnrepository.com/artifact/ai.djl.tensorflow/tensorflow-engine
//    implementation("ai.djl.tensorflow:tensorflow-engine:0.26.0")


    // https://mvnrepository.com/artifact/ai.djl.pytorch/pytorch-engine
//    implementation("ai.djl.pytorch:pytorch-engine:0.26.0")


// https://mvnrepository.com/artifact/ch.qos.logback/logback-classic
    implementation("ch.qos.logback:logback-classic:1.4.14")

}

tasks.getByName<Test>("test") {
    useJUnitPlatform()
}