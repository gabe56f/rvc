import { gql } from "@apollo/client/core";

export const PAST_GENERATIONS = gql`
query PastGenerations($count: Int) {
  pastGenerations(count: $count) {
    createdAt
    id
    inputName
    timeTaken
    modelUsed
    output
    outputCount
    status
  }
}
`;

const example_entry = {
    label: "",
    value: "",
}

export type Entry = typeof example_entry;

export const MODELS = gql`
query Models {
  models {
    label
    value
  }
}
`;

const example_generation = {
    createdAt: "",
    id: 0,
    inputName: "",
    timeTaken: [0.1],
    modelUsed: "",
    output: [""],
    outputCount: 1,
    status: ""
};

export type Generation = typeof example_generation;

const example_preprocess = {
    input: "file",
    type: "MDX23C",
    file: "VR-DeEchoNormal.pth",
    strength: 10,
    saveAccompaniment: true,
    saveVocals: true,
    accompanimentDirectoryOverride: null,
    vocalsDirectoryOverride: null,
}

export type Preprocess = typeof example_preprocess;

export const DEVICES = gql`
query Devices {
  devices {
    index
    memory
    name
    supportedDatatypes
    type
  }
}
`;

export const QUEUE_SUB = gql`
subscription Queue($input: Upload!, $filename: String, $model: String, $transpose: Int, $pitchExtraction: String, $output: String, $preprocess: [Preprocess!]) {
  createAndAwaitResult(
    input: {
      filename: $filename
      filetype: "mp3"
      input: $input
      model: $model
      transpose: $transpose
      pitchExtraction: $pitchExtraction
      preprocessOutput: $output
      preprocess: $preprocess
    }
  ) {
    status
    inputName
    modelUsed
    output
    outputCount
    timeTaken
  }
}
`;

export const QUEUE = gql`
mutation Queue($input: Upload!, $filename: String!, $model: String!, $transpose: Int!, $pitchExtraction: String!, $output: String!, $preprocess: [Preprocess!]!) {
  createAndAwaitResult(
    input: {
      filename: $filename
      filetype: "mp3"
      input: $input
      model: $model
      transpose: $transpose
      pitchExtraction: $pitchExtraction
      preprocessOutput: $output
      preprocess: $preprocess
    }
  ) {
    outputCount
  }
}
`;

export const UVR_MODELS = gql`
query UvrModels {
  uvrModels {
    label
    value
  }
}
`;

export const PITCH_EXTRACTORS = gql`
query PitchExtractors {
  pitchExtractors {
    label
    value
  }
}
`;