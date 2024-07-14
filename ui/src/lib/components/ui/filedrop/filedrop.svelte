<script>
    import { fromEvent } from "file-selector";
    import { onDestroy, createEventDispatcher } from "svelte";

    function accepts(file, acceptedFiles) {
        if (file && acceptedFiles) {
            const acceptedFilesArray = Array.isArray(acceptedFiles)
                ? acceptedFiles
                : acceptedFiles.split(",");
            const fileName = file.name || "";
            const mimeType = (file.type || "").toLowerCase();
            const baseMimeType = mimeType.replace(/\/.*$/, "");

            return acceptedFilesArray.some((type) => {
                const validType = type.trim().toLowerCase();
                if (validType.charAt(0) === ".") {
                    return fileName.toLowerCase().endsWith(validType);
                } else if (validType.endsWith("/*")) {
                    // This is something like a image/* mime type
                    return baseMimeType === validType.replace(/\/.*$/, "");
                }
                return mimeType === validType;
            });
        }
        return true;
    }

    // Error codes
    const FILE_INVALID_TYPE = "file-invalid-type";
    const FILE_TOO_LARGE = "file-too-large";
    const FILE_TOO_SMALL = "file-too-small";
    const TOO_MANY_FILES = "too-many-files";

    // File Errors
    const getInvalidTypeRejectionErr = (accept) => {
        accept =
            Array.isArray(accept) && accept.length === 1 ? accept[0] : accept;
        const messageSuffix = Array.isArray(accept)
            ? `one of ${accept.join(", ")}`
            : accept;
        return {
            code: FILE_INVALID_TYPE,
            message: `File type must be ${messageSuffix}`,
        };
    };

    const getTooLargeRejectionErr = (maxSize) => {
        return {
            code: FILE_TOO_LARGE,
            message: `File is larger than ${maxSize} bytes`,
        };
    };

    const getTooSmallRejectionErr = (minSize) => {
        return {
            code: FILE_TOO_SMALL,
            message: `File is smaller than ${minSize} bytes`,
        };
    };

    const TOO_MANY_FILES_REJECTION = {
        code: TOO_MANY_FILES,
        message: "Too many files",
    };

    // Firefox versions prior to 53 return a bogus MIME type for every file drag, so dragovers with
    // that MIME type will always be accepted
    function fileAccepted(file, accept) {
        const isAcceptable =
            file.type === "application/x-moz-file" || accepts(file, accept);
        return [
            isAcceptable,
            isAcceptable ? null : getInvalidTypeRejectionErr(accept),
        ];
    }

    function fileMatchSize(file, minSize, maxSize) {
        if (isDefined(file.size)) {
            if (isDefined(minSize) && isDefined(maxSize)) {
                if (file.size > maxSize)
                    return [false, getTooLargeRejectionErr(maxSize)];
                if (file.size < minSize)
                    return [false, getTooSmallRejectionErr(minSize)];
            } else if (isDefined(minSize) && file.size < minSize)
                return [false, getTooSmallRejectionErr(minSize)];
            else if (isDefined(maxSize) && file.size > maxSize)
                return [false, getTooLargeRejectionErr(maxSize)];
        }
        return [true, null];
    }

    function isDefined(value) {
        return value !== undefined && value !== null;
    }

    function allFilesAccepted({ files, accept, minSize, maxSize, multiple }) {
        if (!multiple && files.length > 1) {
            return false;
        }

        return files.every((file) => {
            const [accepted] = fileAccepted(file, accept);
            const [sizeMatch] = fileMatchSize(file, minSize, maxSize);
            return accepted && sizeMatch;
        });
    }

    // React's synthetic events has event.isPropagationStopped,
    // but to remain compatibility with other libs (Preact) fall back
    // to check event.cancelBubble
    function isPropagationStopped(event) {
        if (typeof event.isPropagationStopped === "function") {
            return event.isPropagationStopped();
        } else if (typeof event.cancelBubble !== "undefined") {
            return event.cancelBubble;
        }
        return false;
    }

    function isEvtWithFiles(event) {
        if (!event.dataTransfer) {
            return !!event.target && !!event.target.files;
        }
        // https://developer.mozilla.org/en-US/docs/Web/API/DataTransfer/types
        // https://developer.mozilla.org/en-US/docs/Web/API/HTML_Drag_and_Drop_API/Recommended_drag_types#file
        return Array.prototype.some.call(
            event.dataTransfer.types,
            (type) => type === "Files" || type === "application/x-moz-file",
        );
    }

    function isKindFile(item) {
        return (
            typeof item === "object" && item !== null && item.kind === "file"
        );
    }

    function isIe(userAgent) {
        return (
            userAgent.indexOf("MSIE") !== -1 ||
            userAgent.indexOf("Trident/") !== -1
        );
    }

    function isEdge(userAgent) {
        return userAgent.indexOf("Edge/") !== -1;
    }

    function isIeOrEdge(userAgent = window.navigator.userAgent) {
        return isIe(userAgent) || isEdge(userAgent);
    }

    /**
     * This is intended to be used to compose event handlers
     * They are executed in order until one of them calls `event.isPropagationStopped()`.
     * Note that the check is done on the first invoke too,
     * meaning that if propagation was stopped before invoking the fns,
     * no handlers will be executed.
     *
     * @param {Function} fns the event hanlder functions
     * @return {Function} the event handler to add to an element
     */
    function composeEventHandlers(...fns) {
        return (event, ...args) =>
            fns.some((fn) => {
                if (!isPropagationStopped(event) && fn) {
                    fn(event, ...args);
                }
                return isPropagationStopped(event);
            });
    }

    //props
    /**
     * Set accepted file types.
     * See https://github.com/okonet/attr-accept for more information.
     */
    /**
     * @type {string | Array<string>}
     */
    export let accept = undefined;
    export let disabled = false;
    export let getFilesFromEvent = fromEvent;
    export let maxSize = Infinity;
    export let minSize = 0;
    export let multiple = true;
    export let preventDropOnDocument = true;
    export let noClick = false;
    export let noKeyboard = false;
    export let noDrag = false;
    export let noDragEventsBubbling = false;
    export let containerClasses = "";
    export let containerStyles = "";
    export let disableDefaultStyles = false;
    export let name = "";
    export let inputElement = undefined;
    export let required = false;
    const dispatch = createEventDispatcher();

    //state

    let state = {
        isFocused: false,
        isFileDialogActive: false,
        isDragActive: false,
        isDragAccept: false,
        isDragReject: false,
        draggedFiles: [],
        acceptedFiles: [],
        fileRejections: [],
    };

    let rootRef;

    function resetState() {
        state.isFileDialogActive = false;
        state.isDragActive = false;
        state.draggedFiles = [];
        state.acceptedFiles = [];
        state.fileRejections = [];
    }

    // Fn for opening the file dialog programmatically
    function openFileDialog() {
        if (inputElement) {
            inputElement.value = null; // TODO check if null needs to be set
            state.isFileDialogActive = true;
            inputElement.click();
        }
    }

    // Cb to open the file dialog when SPACE/ENTER occurs on the dropzone
    function onKeyDownCb(event) {
        // Ignore keyboard events bubbling up the DOM tree
        if (!rootRef || !rootRef.isEqualNode(event.target)) {
            return;
        }

        if (event.keyCode === 32 || event.keyCode === 13) {
            event.preventDefault();
            openFileDialog();
        }
    }

    // Update focus state for the dropzone
    function onFocusCb() {
        state.isFocused = true;
    }
    function onBlurCb() {
        state.isFocused = false;
    }

    // Cb to open the file dialog when click occurs on the dropzone
    function onClickCb() {
        if (noClick) {
            return;
        }

        // In IE11/Edge the file-browser dialog is blocking, therefore, use setTimeout()
        // to ensure React can handle state changes
        // See: https://github.com/react-dropzone/react-dropzone/issues/450
        if (isIeOrEdge()) {
            setTimeout(openFileDialog, 0);
        } else {
            openFileDialog();
        }
    }

    function onDragEnterCb(event) {
        event.preventDefault();
        stopPropagation(event);

        dragTargetsRef = [...dragTargetsRef, event.target];

        if (isEvtWithFiles(event)) {
            Promise.resolve(getFilesFromEvent(event)).then((draggedFiles) => {
                if (isPropagationStopped(event) && !noDragEventsBubbling) {
                    return;
                }

                state.draggedFiles = draggedFiles;
                state.isDragActive = true;

                dispatch("dragenter", {
                    dragEvent: event,
                });
            });
        }
    }

    function onDragOverCb(event) {
        event.preventDefault();
        stopPropagation(event);

        if (event.dataTransfer) {
            try {
                event.dataTransfer.dropEffect = "copy";
            } catch {} /* eslint-disable-line no-empty */
        }

        if (isEvtWithFiles(event)) {
            dispatch("dragover", {
                dragEvent: event,
            });
        }

        return false;
    }

    function onDragLeaveCb(event) {
        event.preventDefault();
        stopPropagation(event);

        // Only deactivate once the dropzone and all children have been left
        const targets = dragTargetsRef.filter(
            (target) => rootRef && rootRef.contains(target),
        );
        // Make sure to remove a target present multiple times only once
        // (Firefox may fire dragenter/dragleave multiple times on the same element)
        const targetIdx = targets.indexOf(event.target);
        if (targetIdx !== -1) {
            targets.splice(targetIdx, 1);
        }
        dragTargetsRef = targets;
        if (targets.length > 0) {
            return;
        }

        state.isDragActive = false;
        state.draggedFiles = [];

        if (isEvtWithFiles(event)) {
            dispatch("dragleave", {
                dragEvent: event,
            });
        }
    }

    function onDropCb(event) {
        event.preventDefault();
        stopPropagation(event);

        dragTargetsRef = [];

        if (isEvtWithFiles(event)) {
            dispatch("filedropped", {
                event,
            });
            Promise.resolve(getFilesFromEvent(event)).then((files) => {
                if (isPropagationStopped(event) && !noDragEventsBubbling) {
                    return;
                }

                const acceptedFiles = [];
                const fileRejections = [];

                files.forEach((file) => {
                    const [accepted, acceptError] = fileAccepted(file, accept);
                    const [sizeMatch, sizeError] = fileMatchSize(
                        file,
                        minSize,
                        maxSize,
                    );
                    if (accepted && sizeMatch) {
                        acceptedFiles.push(file);
                    } else {
                        const errors = [acceptError, sizeError].filter(
                            (e) => e,
                        );
                        fileRejections.push({ file, errors });
                    }
                });

                if (!multiple && acceptedFiles.length > 1) {
                    // Reject everything and empty accepted files
                    acceptedFiles.forEach((file) => {
                        fileRejections.push({
                            file,
                            errors: [TOO_MANY_FILES_REJECTION],
                        });
                    });
                    acceptedFiles.splice(0);
                }

                // Files dropped keep input in sync
                if (event.dataTransfer) {
                    inputElement.files = event.dataTransfer.files;
                }

                state.acceptedFiles = acceptedFiles;
                state.fileRejections = fileRejections;

                dispatch("drop", {
                    acceptedFiles,
                    fileRejections,
                    event,
                });

                if (fileRejections.length > 0) {
                    dispatch("droprejected", {
                        fileRejections,
                        event,
                    });
                }

                if (acceptedFiles.length > 0) {
                    dispatch("dropaccepted", {
                        acceptedFiles,
                        event,
                    });
                }
            });
        }
        resetState();
    }

    $: composeHandler = (fn) => (disabled ? null : fn);

    $: composeKeyboardHandler = (fn) =>
        noKeyboard ? null : composeHandler(fn);

    $: composeDragHandler = (fn) => (noDrag ? null : composeHandler(fn));

    function stopPropagation(event) {
        if (noDragEventsBubbling) {
            event.stopPropagation();
        }
    }

    // allow the entire document to be a drag target
    function onDocumentDragOver(event) {
        if (preventDropOnDocument) {
            event.preventDefault();
        }
    }

    let dragTargetsRef = [];
    function onDocumentDrop(event) {
        if (!preventDropOnDocument) {
            return;
        }
        if (rootRef && rootRef.contains(event.target)) {
            // If we intercepted an event for our instance, let it propagate down to the instance's onDrop handler
            return;
        }
        event.preventDefault();
        dragTargetsRef = [];
    }

    // Update file dialog active state when the window is focused on
    function onWindowFocus() {
        // Execute the timeout only if the file dialog is opened in the browser
        if (state.isFileDialogActive) {
            setTimeout(() => {
                if (inputElement) {
                    const { files } = inputElement;

                    if (!files.length) {
                        state.isFileDialogActive = false;
                        dispatch("filedialogcancel");
                    }
                }
            }, 300);
        }
    }

    onDestroy(() => {
        // This is critical for canceling the timeout behaviour on `onWindowFocus()`
        inputElement = null;
    });

    function onInputElementClick(event) {
        event.stopPropagation();
    }
</script>

<svelte:window
    on:focus={onWindowFocus}
    on:dragover={onDocumentDragOver}
    on:drop={onDocumentDrop}
/>

<div
    bind:this={rootRef}
    tabindex="0"
    role="button"
    class="{disableDefaultStyles ? '' : 'dropzone'}
    {containerClasses}"
    style={containerStyles}
    on:keydown={composeKeyboardHandler(onKeyDownCb)}
    on:focus={composeKeyboardHandler(onFocusCb)}
    on:blur={composeKeyboardHandler(onBlurCb)}
    on:click={composeHandler(onClickCb)}
    on:dragenter={composeDragHandler(onDragEnterCb)}
    on:dragover={composeDragHandler(onDragOverCb)}
    on:dragleave={composeDragHandler(onDragLeaveCb)}
    on:drop={composeDragHandler(onDropCb)}
>
    <input
        accept={accept?.toString()}
        {multiple}
        {required}
        type="file"
        {name}
        autocomplete="off"
        tabindex="-1"
        on:change={onDropCb}
        on:click={onInputElementClick}
        bind:this={inputElement}
        style="display: none;"
    />
    <slot>
        <p>Drag 'n' drop some files here, or click to select files</p>
    </slot>
</div>

<style>
    .dropzone {
        flex: 1;
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 20px;
        border-width: 2px;
        border-radius: 2px;
        border-color: light-dark(rgba(0, 0, 0, 0.2), rgba(238, 238, 238, 0.2));
        border-style: dashed;
        color: #bdbdbd;
        outline: none;
        transition: border 0.24s ease-in-out;
    }
    .dropzone:focus {
        border-color: #2196f3;
    }
</style>
