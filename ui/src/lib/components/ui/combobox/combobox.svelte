<script lang="ts">
    import Check from "lucide-svelte/icons/check";
    import ChevronsUpDown from "lucide-svelte/icons/chevrons-up-down";
    import * as Command from "$lib/components/ui/command/index.js";
    import * as Popover from "$lib/components/ui/popover/index.js";
    import { Button } from "$lib/components/ui/button/index.js";
    import { cn } from "$lib/utils.js";
    import { onMount, tick } from "svelte";

    export let cclass: string = "w-[60vh] max-w-md";
    export let values: any[] = [];

    export let placeholder: string = "Select a value...";
    export let searchPlaceholder: string = "Search values...";
    export let searchNoResult: string = "No results...";
    export let open = false;
    export let chosen = [""];

    export let multiple: boolean = false;

    onMount(() => {
        if (multiple && chosen[0] == "") chosen = [];
    });

    $: selectedValue = multiple
        ? chosen.length == 0
            ? placeholder
            : chosen.join(", ")
        : values.find((f) => f.value === chosen[0])?.label ?? placeholder;

    // We want to refocus the trigger button when the user selects
    // an item from the list so users can continue navigating the
    // rest of the form with the keyboard.
    function closeAndFocusTrigger(triggerId: string) {
        if (!multiple) open = false;
        tick().then(() => {
            if (!multiple) document.getElementById(triggerId)?.focus();
        });
    }
</script>

<Popover.Root bind:open let:ids>
    <Popover.Trigger asChild let:builder>
        <Button
            builders={[builder]}
            variant="outline"
            role="combobox"
            aria-expanded={open}
            class={cclass + " justify-between"}
        >
            {selectedValue}
            <ChevronsUpDown class="ml-2 h-4 w-4 shrink-0 opacity-50" />
        </Button>
    </Popover.Trigger>
    <Popover.Content class={cclass + " p-0"}>
        <Command.Root>
            <Command.Input placeholder={searchPlaceholder} />
            <Command.Empty>{searchNoResult}</Command.Empty>
            <Command.Group>
                {#each values as entry}
                    <Command.Item
                        value={entry.value}
                        onSelect={(currentValue) => {
                            if (multiple) {
                                let index = chosen.indexOf(currentValue);
                                if (index == -1) chosen.push(currentValue);
                                else chosen.splice(index, 1);
                                chosen = chosen;
                            } else chosen[0] = currentValue;
                            closeAndFocusTrigger(ids.trigger);
                        }}
                    >
                        <Check
                            class={cn(
                                "mr-2 h-4 w-4",
                                chosen.indexOf(entry.value) === -1 &&
                                    "text-transparent",
                            )}
                        />
                        {entry.label}
                    </Command.Item>
                {/each}
            </Command.Group>
        </Command.Root>
    </Popover.Content>
</Popover.Root>
