<script lang="ts">
  import Button from "$lib/components/ui/button/button.svelte";
  import Filedrop from "$lib/components/ui/filedrop/filedrop.svelte";
  import Sidebar from "$lib/components/ui/sidebar/sidebar.svelte";
  import Combobox from "$lib/components/ui/combobox/combobox.svelte";
  import Modetoggle from "$lib/components/ui/button/modetoggle.svelte";
  import Refresh from "$lib/components/ui/button/refresh.svelte";
  import Form from "$lib/components/ui/form/form.svelte";
  import Slider from "$lib/components/ui/slider/slider.svelte";
  import type { ButtonEventHandler } from "bits-ui";
  import ChevronsRight from "lucide-svelte/icons/chevrons-right";
  import X from "lucide-svelte/icons/x";
  import Plus from "lucide-svelte/icons/plus";
  import MoveDown from "lucide-svelte/icons/move-down";
  import MoveUp from "lucide-svelte/icons/move-up";
  import CircleAlert from "lucide-svelte/icons/circle-alert";
  import { Input } from "$lib/components/ui/input";
  import ScrollArea from "$lib/components/ui/scroll-area/scroll-area.svelte";
  import { Toggle } from "$lib/components/ui/toggle";
  import { onMount } from "svelte";
  import { Skeleton } from "$lib/components/ui/skeleton";
  import * as HoverCard from "$lib/components/ui/hover-card";
  import * as Alert from "$lib/components/ui/alert";
  import { swipe } from "svelte-gestures";
  import { ApolloClient, InMemoryCache, HttpLink } from "@apollo/client/core";
  import {
    PAST_GENERATIONS,
    QUEUE,
    MODELS,
    PITCH_EXTRACTORS,
    UVR_MODELS,
    type Generation,
    type Entry,
  } from "./schema";

  const url = window.location.origin;

  const httpLink = new HttpLink({
    uri: url + "/graphql",
    useGETForQueries: true,
  });

  const client = new ApolloClient({
    link: httpLink,
    cache: new InMemoryCache(),
  });

  let dragndrop: HTMLElement;

  let files: any[] = [];
  let fileContent: string;

  let pitchExtractors: Entry[] = [];
  let models: Entry[] = [];
  let uvrModels: Entry[] = [];
  let pastGenerations: Generation[] = [];

  let chosenExtractor = [""];
  let chosenModel = [""];
  let transpose: number[] = [0];
  let indexRate: number[] = [1.0];
  let preprocessOutput = ["file"];
  let preprocess: any[] = [];

  let alertstuff: string[] = [];

  let sidebarOpen = false;
  let swipeTime = Date.now();

  const preprocessInputs = {
    true: [{ label: "File", value: "file" }],
    false: [
      {
        label: "Accompaniment",
        value: "accompaniment",
      },
      {
        label: "Vocal",
        value: "vocals",
      },
    ],
  };

  const separators = [
    {
      label: "MDX23C",
      value: "MDX23C",
    },
    {
      label: "UVR",
      value: "UVR",
    },
    {
      label: "DeepFilterNet",
      value: "DeepFilterNet",
    },
  ];

  $: filename =
    files.length == 0 ? "Drag and drop a file here..." : files[0].name;

  $: preprocessOutputs =
    preprocess.length == 0
      ? [{ label: "File", value: "file" }]
      : [
          {
            label: "Accompaniment",
            value: "accompaniment",
          },
          {
            label: "Vocal",
            value: "vocals",
          },
        ];

  $: trans = transpose[0].toLocaleString("en-US", {
    minimumIntegerDigits: 2,
    useGrouping: false,
  });

  const handle = (e: any) => {
    const { acceptedFiles } = e.detail;
    files = acceptedFiles;
    new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (event) => {
        resolve(event.target?.result as string);
      };
      reader.onerror = reject;
      reader.readAsDataURL(files[0]);
    })
      .then((value) => (fileContent = value as string))
      .catch((err) => console.log(err));
  };

  const click = (e: ButtonEventHandler<MouseEvent>) => {
    alertstuff = [];

    if (files.length == 0) alertstuff.push("No input file provided.");

    if (chosenExtractor.length == 0)
      alertstuff.push("No pitch extractors chosen.");

    if (chosenModel[0] == "") alertstuff.push("No model chosen.");

    for (let j in Object.keys(preprocess)) {
      const value = preprocess[j];
      const i = Number.parseInt(j) + 1;

      if (value.type[0] == "") {
        alertstuff.push(
          "Preprocess #" + i + " doesn't have proper preprocess type set."
        );
        continue;
      }

      if (value.input[0] == "") {
        alertstuff.push("Preprocess #" + i + " doesn't have proper input set.");
        continue;
      }

      if (i == 1 && value.input[0] != "file")
        alertstuff.push("Preprocess #" + i + " doesn't have proper input set.");
      if (i != 1 && value.input[0] == "file")
        alertstuff.push("Preprocess #" + i + " doesn't have proper input set.");
      if (value.type[0] == "UVR" && value.model[0] == "")
        alertstuff.push("Preprocess #" + i + " doesn't have proper model set.");
    }

    if (preprocess.length == 0 && preprocessOutput[0] != "file")
      alertstuff.push("Process input isn't set.");
    if (preprocess.length != 0 && preprocessOutput[0] == "file")
      alertstuff.push("Process input isn't set.");

    if (alertstuff.length == 0) {
      save();

      let preproc = [];
      for (const i in Object.keys(preprocess)) {
        const value = preprocess[i];
        preproc.push({
          input: value.input[0],
          type: value.type[0],
          file: value.model[0],
          saveAccompaniment: value.saveAccompaniment,
          saveVocals: value.saveVocals,
          accompanimentDirectoryOverride: value.accompanimentLocation[0],
          vocalsDirectoryOverride: value.vocalLocation[0],
        });
      }

      (async () => {
        await client.mutate({
          mutation: QUEUE,
          ...{
            variables: {
              input: fileContent,
              filename: filename,
              model: chosenModel[0],
              transpose: transpose[0],
              index: indexRate[0],
              pitchExtraction: chosenExtractor.join(","),
              output: preprocessOutput[0],
              preprocess: preproc,
            },
            http: {
              includeQuery: false,
            },
          },
        });
      })()
        .catch((err) => console.error(err))
        .then(() => refresh());
      sidebarOpen = false;
    }
  };

  let revealed: number[] = [];

  const pauseAudio = (generation: any) => {
    for (let o of Array(generation.outputCount).keys()) {
      const obj = document.querySelector(
        "#audio-" + generation.id + "-" + o + "-obj"
      ) as HTMLAudioElement;
      obj.pause();
    }
  };

  const seekedAudio = (generation: any, i: number) => {
    const now = Date.now();
    if (now - seekTime < 100) return;
    seekTime = now;
    const obj = document.querySelector(
      "#audio-" + generation.id + "-" + i + "-obj"
    ) as HTMLAudioElement;
    const time = obj.currentTime;
    for (let o of Array(generation.outputCount).keys()) {
      const obj = document.querySelector(
        "#audio-" + generation.id + "-" + o + "-obj"
      ) as HTMLAudioElement;
      obj.currentTime = time;
    }
  };

  const playAudio = (generation: any) => {
    for (let o of Array(generation.outputCount).keys()) {
      const obj = document.querySelector(
        "#audio-" + generation.id + "-" + o + "-obj"
      ) as HTMLAudioElement;
      obj.play();
    }
  };

  let seekTime = Date.now();

  const setAudio = (generation: any) => {
    if (generation.status == "done") {
      for (let o of Array(generation.outputCount).keys()) {
        const skeleton = document.querySelector(
          "#audio-" + generation.id + "-" + o + "-skeleton"
        ) as HTMLElement;
        const obj = document.querySelector(
          "#audio-" + generation.id + "-" + o + "-obj"
        ) as HTMLElement;

        skeleton.style.display = "none";
        (obj.parentElement as HTMLElement).style.display = "block";
        // (obj as HTMLAudioElement).src = generation.output[o];
      }
      revealed.push(generation.id);
    }
  };

  const refresh = () => {
    client
      .query({
        query: PAST_GENERATIONS,
        variables: { count: 10 },
        fetchPolicy: "no-cache",
      })
      .then((value) => {
        pastGenerations = value.data.pastGenerations;
      });
    client.query({ query: MODELS, fetchPolicy: "no-cache" }).then((value) => {
      models = value.data.models;
    });
    client
      .query({ query: UVR_MODELS, fetchPolicy: "no-cache" })
      .then((value) => {
        uvrModels = value.data.uvrModels;
      });
  };

  const save = () => {
    localStorage.setItem("extractor", JSON.stringify(chosenExtractor));
    localStorage.setItem("model", JSON.stringify(chosenModel));
    localStorage.setItem("transpose", JSON.stringify(transpose));
    localStorage.setItem("indexRate", JSON.stringify(indexRate));
    localStorage.setItem("preprocessOutput", JSON.stringify(preprocessOutput));
    localStorage.setItem("preprocess", JSON.stringify(preprocess));
  };

  const load = () => {
    chosenExtractor = JSON.parse(localStorage.getItem("extractor") || "[]");
    chosenModel = JSON.parse(localStorage.getItem("model") || "[]");
    transpose = JSON.parse(localStorage.getItem("transpose") || "[0]");
    indexRate = JSON.parse(localStorage.getItem("indexRate") || "[1.0]");
    preprocessOutput = JSON.parse(
      localStorage.getItem("preprocessOutput") || "[]"
    );
    preprocess = JSON.parse(localStorage.getItem("preprocess") || "[]");
  };

  onMount(() => {
    // Pitch extractors won't update with the exception of RMVPE, but who cares...
    client.query({ query: PITCH_EXTRACTORS }).then((value) => {
      if (!value.loading) pitchExtractors = value.data.pitchExtractors;
    });

    dragndrop = document.querySelector("#dragndrop") as HTMLElement;
    refresh();
    load();
  });
</script>

<div
  class="w-screen h-screen overflow-hidden"
  use:swipe={{ timeframe: 300, minSwipeDistance: 100, touchAction: "pan-y" }}
  on:swipe={(event) => {
    const direction = event.detail.direction;
    sidebarOpen = direction == "right";
    swipeTime = Date.now();
  }}
>
  <Modetoggle />
  <Refresh {refresh} />
  <!-- svelte-ignore a11y-click-events-have-key-events -->
  <!-- svelte-ignore a11y-no-static-element-interactions -->
  <div
    class="absolute h-screen w-screen z-10 select-none"
    on:click={() => {
      const now = Date.now();
      if (now - swipeTime < 50) swipeTime = now;
      else if (sidebarOpen) sidebarOpen = false;
    }}
    on:dblclick={() => {
      sidebarOpen = true;
    }}
    on:dragover={(event) => {
      event.preventDefault();
      dragndrop.style.display = "flex";
    }}
    on:drop={(event) => {
      event.preventDefault();
      const ev = { detail: { acceptedFiles: [] } };
      if (event.dataTransfer?.items)
        ev.detail.acceptedFiles = [event.dataTransfer.items[0].getAsFile()];
      else ev.detail.acceptedFiles = [event.dataTransfer?.files[0]];

      handle(ev);
      sidebarOpen = true;
      dragndrop.style.display = "none";
    }}
  >
    <!-- Past generations -->
    <ScrollArea class="h-screen w-full absolute p-12">
      <div class="space-y-6">
        {#each pastGenerations as generation}
          <div
            class="relative bg-slate-300 dark:bg-gray-900 rounded-xl w-full h-auto py-2 px-6"
          >
            <h1 class="text-2xl">
              {generation.modelUsed}
              <h3 class="text-xl inline-block text-gray-400">
                - {generation.inputName}
              </h3>
            </h1>
            <div class="absolute right-4 top-0 space-x-12">
              <div class="inline-block">
                <HoverCard.Root>
                  <HoverCard.Trigger
                    ><p class="text-sm text-gray-500">
                      Time taken: {generation.timeTaken
                        .reduce((partialSum, a) => partialSum + a, 0)
                        .toLocaleString("en-US", {
                          maximumFractionDigits: 2,
                          useGrouping: false,
                        })}s
                    </p></HoverCard.Trigger
                  >
                  <HoverCard.Content>
                    <ol>
                      {#each generation.timeTaken as tt}
                        <li class="text-gray-500">
                          {(tt * 1000).toLocaleString("en-US", {
                            maximumFractionDigits: 2,
                            useGrouping: false,
                          })}ms
                        </li>
                      {/each}
                    </ol>
                  </HoverCard.Content>
                </HoverCard.Root>
              </div>
              <h4 class="inline-block text-lg text-gray-500">
                #{generation.id}
              </h4>
            </div>
            {#each { length: generation.outputCount } as _, i}
              <div
                id={"audio-" + generation.id + "-" + i + "-skeleton"}
                class="rounded-full w-full h-[60px] py-2"
                on:mouseenter={() => setAudio(generation)}
              >
                <Skeleton class="w-full h-full rounded-full" />
              </div>
              <div style="display: none;" class="w-full h-[60px] py-2">
                <audio
                  controls
                  src={generation.output[i]}
                  id={"audio-" + generation.id + "-" + i + "-obj"}
                  class="w-full"
                  on:play={() => playAudio(generation)}
                  on:pause={() => pauseAudio(generation)}
                  on:seeked={() => seekedAudio(generation, i)}
                />
              </div>
            {/each}
          </div>
        {/each}
      </div>
    </ScrollArea>

    <!-- Drag and drop overlay -->
    <div
      id="dragndrop"
      style="display: none;"
      class="h-screen bg-opacity-25 backdrop-blur-lg bg-slate-500 z-[998]"
      on:dragleave={() => {
        dragndrop.style.display = "none";
      }}
    >
      <div class="m-auto">
        <h1 class="text-8xl text-center p-4">🗃️</h1>
        <h3 class="text-3xl text-center text-slate-600 drop-shadow-lg">
          Drag'n drop files here...
        </h3>
      </div>
    </div>
  </div>
  <Sidebar open={sidebarOpen}>
    <div slot="mobile">
      <div class="z-[999] fixed right-5 top-5">
        <Button
          size="icon"
          variant="outline"
          on:click={() => {
            sidebarOpen = false;
          }}
        >
          <X />
        </Button>
      </div>
    </div>
    <slot>
      <Form label="File input">
        <div slot="main">
          <Filedrop on:drop={handle}>
            <p class="drop-shadow-md text-slate-600 dark:text-gray-200">
              {filename}
            </p>
          </Filedrop>
        </div>
        <div slot="hover">
          <h4 class="text-lg font-semibold drop-shadow-md text-slate-600">
            Input file
          </h4>
          <p class="text-sm inline">
            Can be <code
              class="relative rounded bg-muted px-[0.3rem] py-[0.2rem] font-mono text-sm"
              >audio</code
            >
            or
            <code
              class="relative rounded bg-muted px-[0.3rem] py-[0.2rem] font-mono text-sm"
              >video</code
            >
          </p>
        </div>
      </Form>

      <Form label="Pitch extractors">
        <div slot="main">
          <Combobox
            multiple
            bind:values={pitchExtractors}
            placeholder={"Select a pitch extractor..."}
            bind:chosen={chosenExtractor}
          />
        </div>
      </Form>

      <Form label="Model">
        <div slot="main">
          <Combobox
            bind:values={models}
            placeholder={"Select a model..."}
            bind:chosen={chosenModel}
          />
        </div>
      </Form>

      <Form label="Transpose">
        <div slot="main" class="flex gap-3 h-auto align-center">
          <p class="text-xs font-mono">-12</p>
          <Slider min={-12} max={12} bind:value={transpose} />
          <p class="text-xs font-mono">12</p>
          <p />
          <p class="text-xn font-mono bg-muted rounded px-[0.3rem] py-[0.2rem]">
            {trans}
          </p>
        </div>
        <div slot="hover">
          <h4 class="text-lg font-semibold drop-shadow-md text-slate-600">
            Transpose
          </h4>
          <p class="text-sm inline">
            How much to offset the detected pitch by. <code
              class="relative rounded bg-muted px-[0.3rem] py-[0.2rem] font-mono text-sm"
              >-12</code
            >
            is down an octave, and
            <code
              class="relative rounded bg-muted px-[0.3rem] py-[0.2rem] font-mono text-sm"
              >+12</code
            > is up one.
          </p>
        </div>
      </Form>

      <Form label="Index rate">
        <div slot="main" class="flex gap-3 h-auto align-center">
          <p class="text-xs font-mono">0</p>
          <Slider min={0} max={1} step={0.1} bind:value={indexRate} />
          <p class="text-xs font-mono">1</p>
          <p />
          <p class="text-xn font-mono bg-muted rounded px-[0.3rem] py-[0.2rem]">
            {indexRate}
          </p>
        </div>
        <div slot="hover">
          <h4 class="text-lg font-semibold drop-shadow-md text-slate-600">
            Index rate
          </h4>
          <p class="text-sm inline">
            A higher index rate <i>(in the presence of an index file)</i> will usually
            result in more accurate accent reproduction.
          </p>
        </div>
      </Form>

      <Form label="Preprocess">
        <ScrollArea
          slot="main"
          class="bg-muted border-2 border-dashed rounded-lg h-72 max-h-72 relative"
        >
          <div class="z-40 absolute bottom-0 p-4 right-0">
            <Button
              size="icon"
              variant="outline"
              on:click={() => {
                preprocess.push({
                  type: [""],
                  input: [preprocess.length == 0 ? "file" : ""],
                  model: [""],
                  saveVocals: false,
                  saveAccompaniment: false,
                  accompanimentLocation: ["accompaniment/"],
                  vocalLocation: ["vocals/"],
                });
                preprocess = preprocess;
              }}><Plus /></Button
            >
          </div>
          <div class="p-3 grid grid-rows-1 gap-1">
            {#each preprocess as preprocessentry, i}
              <div class="bg-background rounded-md p-2 relative">
                <div
                  class="inline-flex flex-row align-center justify-center h-[40px] gap-2"
                >
                  <Combobox
                    cclass="max-w-64"
                    values={preprocessInputs[i == 0]}
                    placeholder="Select an input..."
                    bind:chosen={preprocessentry.input}
                  />
                  <ChevronsRight class="opacity-50" />
                  <Combobox
                    cclass="max-w-32"
                    values={separators}
                    placeholder="Select a model..."
                    bind:chosen={preprocessentry.type}
                  />
                </div>
                <div class="absolute right-2.5 top-2.5">
                  <Button
                    disabled={i == 0}
                    variant="outline"
                    size="icon"
                    on:click={() => {
                      const old = preprocess[i - 1];
                      if (i == 1) {
                        preprocessentry.input = ["file"];
                        old.input = ["vocals"];
                      }
                      preprocess[i - 1] = preprocessentry;
                      preprocess[i] = old;
                    }}
                  >
                    <MoveUp size="16" />
                  </Button>
                </div>
                <div class="absolute right-16 top-2.5">
                  <Button
                    disabled={preprocess.length == i + 1}
                    variant="outline"
                    size="icon"
                    on:click={() => {
                      const old = preprocess[i + 1];
                      if (i == 0) {
                        old.input = ["file"];
                        preprocessentry.input = ["vocals"];
                      }
                      preprocess[i + 1] = preprocessentry;
                      preprocess[i] = old;
                    }}
                  >
                    <MoveDown size="16" />
                  </Button>
                </div>
                <div class="absolute right-[7.375rem] top-2.5">
                  <Button
                    variant="outline"
                    size="icon"
                    on:click={() => {
                      preprocess.splice(i, 1);
                      preprocess = preprocess;
                    }}
                  >
                    <X size="16" />
                  </Button>
                </div>
                {#if preprocessentry.type == "UVR"}
                  <Form label="Model">
                    <div slot="main">
                      <Combobox
                        bind:values={uvrModels}
                        cclass="max-w-48"
                        placeholder="Select a model..."
                        bind:chosen={preprocessentry.model}
                      />
                    </div>
                  </Form>
                {/if}
                <div class="inline-flex w-full px-4 py-2 gap-2">
                  <Toggle
                    variant="outline"
                    aria-label="Toggle vocal"
                    bind:pressed={preprocessentry.saveVocals}
                    >Save vocals</Toggle
                  >
                  {#if preprocessentry.saveVocals}
                    <p class="text-sm">Save location:</p>
                    <Input bind:value={preprocessentry.vocalLocation} />
                  {/if}
                </div>
                {#if preprocessentry.type != "DeepFilterNet"}
                  <div class="inline-flex w-full px-4 py-2 gap-2">
                    <Toggle
                      variant="outline"
                      aria-label="Toggle accompaniment"
                      bind:pressed={preprocessentry.saveAccompaniment}
                      >Save accompaniment</Toggle
                    >
                    {#if preprocessentry.saveAccompaniment}
                      <p class="text-sm">Save location:</p>
                      <Input
                        bind:value={preprocessentry.accompanimentLocation}
                      />
                    {/if}
                  </div>
                {/if}
              </div>
            {/each}
          </div>
        </ScrollArea>
      </Form>

      <Form label="Process input">
        <div slot="main">
          <Combobox
            bind:values={preprocessOutputs}
            placeholder={"Select an output..."}
            bind:chosen={preprocessOutput}
          />
        </div>
      </Form>

      {#if alertstuff.length != 0}
        <div class="w-full p-5">
          <Alert.Root variant="destructive">
            <CircleAlert class="h-4 w-4" />
            <Alert.Title>Error</Alert.Title>
            <Alert.Description>
              The following problems need to be fixed before inference can
              begin:
              <ul class="my-6 ml-6 list-disc [&>li]:mt-2">
                {#each alertstuff as af}
                  <li>{af}</li>
                {/each}
              </ul>
            </Alert.Description>
          </Alert.Root>
        </div>
      {/if}

      <div class="absolute z-50 w-full bottom-5 grid grid-cols-3">
        <div />
        <Button size="lg" on:click={click}>Submit</Button>
      </div>
    </slot>
  </Sidebar>
</div>

<style>
  .align-center {
    align-items: center;
  }
</style>
